# Adaptações do BeamQA neste projeto

Este documento registra o que foi efetivamente modificado em relação ao repositório
original ([colab-nyuad/BeamQA](https://github.com/colab-nyuad/BeamQA), disponível
localmente em `methods/_original_reference/BeamQA/` para comparação direta).
Gerado por análise de diff real (ignorando diferenças de fim de linha CRLF/LF).

## Resumo executivo

Duas mudanças independentes:
1. **Troca da fonte do KG embedding (KGE)**: de checkpoints treinados via LibKGE para
   checkpoints treinados via PyKEEN, usados **congelados** (sem re-treinar os pesos do KGE
   junto com a tarefa de QA — o modo `train-BeamQA` do original faz esse fine-tuning, e foi
   deliberadamente evitado por risco de overfitting da embedding ao conjunto de treino de QA).
2. **Rastreamento de caminho ponta a ponta**: o algoritmo de beam search original só devolvia
   a melhor entidade final; a versão modificada carrega a lista de triplas `(head, relation, tail)`
   percorridas em cada candidato, permitindo montar o mesmo tipo de evidência estruturada que o
   TransferNet passou a expor, para alimentar o módulo final de LLM.

Feito **depois** da adaptação do TransferNet — o padrão do módulo de LLM final (`Modulo2/`) foi
reaproveitado, mas com mais tratamento de casos (hits vs. "late hits", perguntas sem candidatos).

## Pastas e o que cada uma é

| Pasta/arquivo | Status | Descrição |
|---|---|---|
| `ModeloGeral/Model.py` | **Não modificada** | A rede ComplEx de scoring (`encoder_head`, `encoder_rel`, operação `ComplEx`) é idêntica ao original. A mudança não é na arquitetura de scoring. |
| `ModeloGeral/pykeen_loader.py` | **Novo — sem equivalente no original** | Substitui `loaders.py` (original, baseado em LibKGE: `KgeModel`, `checkpoint_best.pt`, `entity_ids.del`). Carrega um checkpoint treinado via **PyKEEN** (`trained_model.pkl`, `entity_to_id.tsv.gz`/`relation_to_id.tsv.gz`), extraindo `entity_representations[0]()` e concatenando parte real/imaginária do ComplEx manualmente. `embedding_dim` default mudou de 400 (original) para 512. |
| `ModeloGeral/beamQA.py` | **Modificada substancialmente** (~340 linhas de diferença real) | `check()`: agora retorna `(entidade, score, tripla)` em vez de só `(entidade, score)` — rastreia qual tripla do KG originou cada candidato. Também corrige uma lacuna do original: se a relação não tiver nenhuma aresta de saída válida (`edgeidx` vazio), o original ainda rodava `topk` sobre scores não filtrados; a versão nova retorna `(None, None, None)` nesse caso e usa máscara `-1e9` explícita antes do `topk` (mais seguro que o `index_fill_` do original). `check_rec()`: carrega a lista acumulada de triplas por candidato (`prev_triples + [triple]`) a cada hop. **Nova função `path_finder_candidates()`** (não existe no original): em vez de devolver só o melhor caminho (`path_finder_rec`, mantida como legado), devolve **todos os candidatos finais** com suas triplas e scores — é esse campo (`candidate_paths`) que o `Modulo2/` consome. |
| `ModeloGeral/train_eval.py` | **Modificada** | `evaluate_beamQA()` agora recebe também `triples` de `path_finder_rec()` (função antiga, mantida). Contém um bloco antigo comentado como referência histórica. |
| `ModeloGeral/main.py` | **Modificada** | Import trocado de `loaders` → `pykeen_loader`; paths de KG apontando para os artefatos do PyKEEN (`.pkl` em vez de LibKGE); `embedding_dim` 400→512. Ainda mantém o modo `train-BeamQA` (fine-tuning), mas esse fluxo não é o utilizado. |
| `ModeloGeral/evaluate.py` | **Novo — sem equivalente no original** | É o pipeline atual de ponta a ponta: `paths_generator_modulo1` (gera caminhos candidatos via BART) → `path_finder_candidates` → `predict_answer`. **Atenção**: a linha 6 tem um import com path absoluto estranho (`from Métodos.BeamQA.ModeloGeral.paths_generator_modulo1 import ...`) que provavelmente só funciona se o script for executado de um diretório raiz específico — vale revisar/trocar por um import relativo (`from paths_generator_modulo1 import ...`) para evitar quebra ao rodar de outro lugar. |
| `ModeloGeral/predict_answer.py` | **Novo — sem equivalente no original** | Função pequena e independente: escolhe o candidato de maior `beam_score * path_score` dentre os retornados por `path_finder_candidates()`. É o "modo sem LLM" — útil como baseline de comparação contra o `Modulo2/`. |
| `ModeloGeral/load_graph.py`, `ModeloGeral/utils.py`, `ModeloGeral/teste_simples.py` | Não conferidos em detalhe nesta rodada | `teste_simples.py` parece ser um script de teste manual/scratch — candidato a mover para fora de `ModeloGeral/` ou remover se não for mais usado. |
| `Path_generation/` | Refatorada, comportamento equivalente ao original | Reorganizada em funções reutilizáveis (`generate_paths()`, `load_model()`), com filtro opcional de relações válidas do MetaQA. Não é uma mudança de comportamento central, é um refactor de organização. |
| `Modulo2/rog_llama_predict_beamQA.py` | **Novo — sem equivalente no original** | Mesmo esqueleto do módulo de LLM do TransferNet (carrega LLM causal, monta prompt, gera, faz parsing, mede métricas), mas lê `candidate_paths` (formato do `path_finder_candidates()`) em vez de `hop_N`. Mais maduro que a versão do TransferNet: separa "hits1" de "late_hit" (resposta certa aparece na lista mas não em 1º lugar) e contabiliza `empty_candidate_paths` separadamente. |
| `Data/` | Dados, não código | Não copiado para `_original_reference/` (é bruto, ~87 MB no original). |

## Pontos de atenção para limpeza futura

- Corrigir o import de path absoluto em `evaluate.py` linha 6.
- `path_finder_rec()` (versão antiga, só melhor caminho) ficou como código morto depois de `path_finder_candidates()` — decidir se remove ou mantém como fallback documentado.
- `teste_simples.py` — confirmar se ainda é necessário ou pode sair de `ModeloGeral/`.
- O modo `train-BeamQA` em `main.py` (fine-tuning do KGE) continua no código mas não é o fluxo usado — vale um comentário no próprio arquivo explicando por que foi descontinuado (overfitting), para quem ler o código depois não reativar sem contexto.
