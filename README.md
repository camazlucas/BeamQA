# BeamQA

Implementation of "Question Answering over Incomplete Knowledge Graphs using Path Prediction and Graph Embeddings"


# Instructions 
### Quick start
```sh
# retrieve and install project 
git clone https://github.com/colab-nyuad/BeamQA

# install requirements
pip install -r requirements.txt
git clone https://github.com/uma-pi1/kge.git
cd kge
pip install -e .
```

## Data 

We use two datasets MetaQA and WebQuestionSP
The Data folder can be downloaded from [this link](https://drive.google.com/file/d/1oEDSK2e1R67L1fee4YhxGnrwkArSe162/view?usp=sharing).*  

#### Graph emeddings 
We generate graph embeddings using Libkge, the graphs and config files are provided. Further instructions on how to train embeddings can be found in LibkGE repository https://github.com/uma-pi1/kge


#### Train-eval MetaQA : 
```sh
cd BeamQA/MetaQA 
python main.py --gpu [number] --kg_type half --mode train-BeamQA
```
Or 
```sh
bash train.sh
```


#### Train-eval WQSP : 
```sh
cd BeamQA/WQSP
python main.py --gpu [number] --hops [num_hops] --kg_type [kg_type] --mode train-BeamQA
```
Or uncomment lines for WQSP
```sh
bash train.sh
```

### Path generation 
Code for generating paths and Synthetic questions can be found in BeamQA/Path_generation


## Results

## How to cite


##### Métricas para o MetaQA

 python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/MetaQA/test_1hop.txt --output_file Data/Path_gen/outputs/path_generation_metaqa_1hop.txt --model_path AtualizacaoMetaQA/Modulo1/bart_metaqa_all_hops/final_model/ --use_relation_filter --kg_model_path Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv --graph_file Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl --model_type bart

 python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/MetaQA/test_2hop.txt --output_file Data/Path_gen/outputs/path_generation_metaqa_2hop.txt --model_path AtualizacaoMetaQA/Modulo1/bart_metaqa_all_hops/final_model/ --use_relation_filter --kg_model_path Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv --graph_file Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl --model_type bart
 
 python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/MetaQA/test_3hop.txt --output_file Data/Path_gen/outputs/path_generation_metaqa_3hop.txt --model_path AtualizacaoMetaQA/Modulo1/bart_metaqa_all_hops/final_model/ --use_relation_filter --kg_model_path Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv --graph_file Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl --model_type bart


##### Gerador de JSON

 python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/MetaQA/test_1hop.txt --output_file Data/Path_gen/outputs/path_generation_metaqa_1hop.json --model_path AtualizacaoMetaQA/Modulo1/bart_metaqa_all_hops/final_model/ --use_relation_filter --kg_model_path Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv --graph_file Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl --model_type bart

 python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/MetaQA/test_2hop.txt --output_file Data/Path_gen/outputs/path_generation_metaqa_2hop.json --model_path AtualizacaoMetaQA/Modulo1/bart_metaqa_all_hops/final_model/ --use_relation_filter --kg_model_path Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv --graph_file Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl --model_type bart

  python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/MetaQA/test_3hop.txt --output_file Data/Path_gen/outputs/path_generation_metaqa_3hop.json --model_path AtualizacaoMetaQA/Modulo1/bart_metaqa_all_hops/final_model/ --use_relation_filter --kg_model_path Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv --graph_file Data/Graph_data/MetaQA/MetaQAProcess/new_complex_metaqa_100_inv/MetaQA_graph.pkl --model_type bart

  python AtualizacaoMetaQA/evaluate.py --test_file Data/QA_data/WQSP/test_wqsp.txt --output_file Data/Path_gen/outputs/path_generation_webqsp.json --model_path "WQSP/Modulo 1/Modelo1-BART/final_model/" --kg_model_path Data/Graph_data/FreeBase/complex_freebase_64_inv/ --graph_file Data/Graph_data/FreeBase/freebase_nxgraph.pkl --model_type bart


#### Predicao com o Llama BeamQA

python "Modulo2/rog_llama_predict_beamQA.py" --model_path "../../Dataset/LLM/RoG/" --json_path Data/Path_gen/outputs/sample.json --output_path "../../Resultados/BeamQALLM/sample.json"

python "Modulo2/rog_llama_predict_beamQA.py" --model_path "../../Dataset/LLM/RoG/" --json_path Data/Path_gen/outputs/path_generation_metaqa_1hop.json --output_path "../../Resultados/BeamQALLM/metaqa_1hop.json"

python "Modulo2/rog_llama_predict_beamQA.py" --model_path "../../Dataset/LLM/RoG/" --json_path Data/Path_gen/outputs/path_generation_metaqa_2hop.json --output_path "../../Resultados/BeamQALLM/metaqa_2hop.json"

python "Modulo2/rog_llama_predict_beamQA.py" --model_path "../../Dataset/LLM/RoG/" --json_path Data/Path_gen/outputs/path_generation_metaqa_3hop.json --output_path "../../Resultados/BeamQALLM/metaqa_3hop.json"

python "Modulo2/rog_llama_predict_beamQA.py" --model_path "../../Dataset/LLM/RoG/" --json_path Data/Path_gen/outputs/path_generation_wqsp.json --output_path "../../Resultados/BeamQALLM/wqsp.json"