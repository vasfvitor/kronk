package kronk

import (
	"context"
	"fmt"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
)

type model struct {
	cfg       ModelConfig
	model     llama.Model
	vocab     llama.Vocab
	ctxParams llama.ContextParams
	template  string
	projFile  string
	muHEC     sync.Mutex
}

func newModel(modelFile string, cfg ModelConfig, options ...func(m *model) error) (*model, error) {
	mdl, err := llama.ModelLoadFromFile(modelFile, llama.ModelDefaultParams())
	if err != nil {
		return nil, fmt.Errorf("ModelLoadFromFile: %w", err)
	}

	cfg = adjustConfig(cfg, mdl)

	vocab := llama.ModelGetVocab(mdl)

	// -------------------------------------------------------------------------

	template := llama.ModelChatTemplate(mdl, "")
	if template == "" {
		template, _ = llama.ModelMetaValStr(mdl, "tokenizer.chat_template")
	}

	if template == "" {
		template = "chatml"
	}

	// -------------------------------------------------------------------------

	m := model{
		cfg:       cfg,
		model:     mdl,
		vocab:     vocab,
		ctxParams: modelCtxParams(cfg),
		template:  template,
	}

	for _, option := range options {
		if err := option(&m); err != nil {
			llama.ModelFree(mdl)
			llama.BackendFree()
			return nil, err
		}
	}

	return &m, nil
}

func (m *model) unload() {
	llama.ModelFree(m.model)
	llama.BackendFree()
}

func (m *model) modelInfo() ModelInfo {
	desc := llama.ModelDesc(m.model)
	size := llama.ModelSize(m.model)
	encoder := llama.ModelHasEncoder(m.model)
	decoder := llama.ModelHasDecoder(m.model)
	recurrent := llama.ModelIsRecurrent(m.model)
	hybrid := llama.ModelIsHybrid(m.model)
	count := llama.ModelMetaCount(m.model)
	metadata := make(map[string]string)

	for i := range count {
		key, ok := llama.ModelMetaKeyByIndex(m.model, i)
		if !ok {
			continue
		}

		value, ok := llama.ModelMetaValStrByIndex(m.model, i)
		if !ok {
			continue
		}

		metadata[key] = value
	}

	return ModelInfo{
		Desc:        desc,
		Size:        size,
		HasEncoder:  encoder,
		HasDecoder:  decoder,
		IsRecurrent: recurrent,
		IsHybrid:    hybrid,
		Metadata:    metadata,
	}
}

func (m *model) processTokens(ctx context.Context, startingTokens []llama.Token, lctx llama.Context, sampler llama.Sampler, ch chan<- ChatResponse) {
	var outputTokens int
	var contextTokens int

	tokens := startingTokens

	const bufferSize = 32 * 1024
	buf := make([]byte, bufferSize)

	batch := llama.BatchGetOne(tokens)
	inputTokens := int(batch.NTokens)
	contextTokens += inputTokens

	for outputTokens < m.cfg.MaxTokens {
		select {
		case <-ctx.Done():
			ch <- ChatResponse{
				Err: ctx.Err(),
				Tokens: Tokens{
					Input:   inputTokens,
					Output:  outputTokens,
					Context: contextTokens,
				},
			}
			return
		default:
		}

		llama.Decode(lctx, batch)
		token := llama.SamplerSample(sampler, lctx, -1)

		if llama.VocabIsEOG(m.vocab, token) {
			break
		}

		l := llama.TokenToPiece(m.vocab, token, buf, 0, false)

		resp := string(buf[:l])
		if resp == "" {
			break
		}

		select {
		case <-ctx.Done():
			ch <- ChatResponse{
				Err: ctx.Err(),
				Tokens: Tokens{
					Input:   inputTokens,
					Output:  outputTokens,
					Context: contextTokens,
				},
			}
			return

		case ch <- ChatResponse{
			Response: resp,
			Tokens: Tokens{
				Input:   inputTokens,
				Output:  outputTokens,
				Context: contextTokens,
			}}:
		}

		tokens = []llama.Token{token}
		batch = llama.BatchGetOne(tokens)

		outputTokens = int(batch.NTokens)
		contextTokens += outputTokens
	}
}
