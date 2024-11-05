# SLM Survey

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![](https://img.shields.io/badge/PaperNumber-152-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red) 


## A Comprehensive Survey of Small Language Models: Technology, On-Device Applications, Efficiency, Enhancements for LLMs, and Trustworthiness

## Overview of SLMs
![Overview of Small Language Models](images/overview_structure.png)

## Timeline of SLMs
![Timeline of Small Language Models](images/timeline.png)

## SLMs Paper List
### Existing SLMs


| Model | #Params | Date | Paradigm | Domain | Code | HF Model | Paper/Blog |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |---- |
| Llama 3.2 | 1B; 3B | 2024.9 | Pre-train | Generic | [Github](https://github.com/meta-llama/llama-models)  | [HF](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | [Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| Qwen 1 | 1.8B; 7B; 14B; 72B | 2023.12 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2309.16609) |
| Qwen 1.5 | 0.5B; 1.8B; 4B; 7B; 14B; 32B; 72B | 2024.2 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2309.16609) |
| Qwen 2 | 0.5B; 1.5B; 7B; 57B; 72B | 2024.6 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2407.10671) |
| Qwen 2.5 | 0.5B; 1.5B; 3B; 7B; 14B; 32B; 72B | 2024.9 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2407.10671) |
| Gemma | 2B; 7B | 2024.2 | Pre-train | Generic | | [HF](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b) | [Paper](https://arxiv.org/abs/2403.08295) |
| Gemma 2 | 2B; 9B; 27B | 2024.7 | Pre-train | Generic | | [HF](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) | [Paper](https://arxiv.org/abs/2408.00118) |
| H2O-Danube3 | 500M; 4B | 2024.7 | Pre-train | Generic | | [HF](https://huggingface.co/collections/h2oai/h2o-danube3-6687a993641452457854c609) | [Paper](https://arxiv.org/abs/2407.09276) |
| Fox-1 | 1.6B | 2024.6 | Pre-train | Generic | | [HF](https://huggingface.co/tensoropera/Fox-1-1.6B) | [Blog](https://blog.tensoropera.ai/tensoropera-unveils-fox-foundation-model-a-pioneering-open-source-slm-leading-the-way-against-tech-giants/) |
| Rene | 1.3B | 2024.5 | Pre-train | Generic | | [HF](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch) | [Paper](https://cartesia.ai/blog/on-device) |
| MiniCPM | 1.2B; 2.4B | 2024.4 | Pre-train | Generic | [Github](https://github.com/OpenBMB/MiniCPM-V) | [HF](https://huggingface.co/collections/openbmb/minicpm-65d48bf958302b9fd25b698f) | [Paper](https://arxiv.org/abs/2404.06395) |
| OLMo | 1B; 7B | 2024.2 | Pre-train | Generic| [Github](https://github.com/allenai/OLMo)   | [HF](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778) | [Paper](https://arxiv.org/abs/2402.00838) |
| TinyLlama | 1B | 2024.1 | Pre-train | Generic| [Github](https://github.com/jzhang38/TinyLlama) | [HF](https://huggingface.co/TinyLlama) | [Paper](https://arxiv.org/abs/2401.02385) |
| Phi-1 | 1.3B | 2023.6 | Pre-train | Coding | | [HF](https://huggingface.co/microsoft/phi-1) | [Paper](https://arxiv.org/abs/2306.11644) |
| Phi-1.5 | 1.3B | 2023.9 | Pre-train | Generic | | [HF](https://huggingface.co/microsoft/phi-1_5) | [Paper](https://arxiv.org/abs/2309.05463) |
| Phi-2 | 2.7B | 2023.12 | Pre-train | Generic | | [HF](https://huggingface.co/microsoft/phi-2) | [Paper](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) |
| Phi-3 | 3.8B; 7B; 14B | 2024.4 | Pre-train | Generic | | [HF](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | [Paper](https://arxiv.org/abs/2404.14219) |
| Phi-3.5 | 3.8B; 4.2B; 6.6B | 2024.4 | Pre-train | Generic | | [HF](https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3) | [Paper](https://arxiv.org/abs/2404.14219) |
| OpenELM | 270M; 450M; 1.1B; 3B | 2024.4 | Pre-train | Generic|  [Github](https://github.com/CarperAI/OpenELM) | [HF](https://huggingface.co/apple/OpenELM) | [Paper](https://openreview.net/forum?id=XNMbTkxroF) |
| MobiLlama | 0.5B; 0.8B | 2024.2 | Pre-train | Generic | [Github](https://github.com/mbzuai-oryx/MobiLlama)  | [HF](https://huggingface.co/apple/OpenELM) | [Paper](URL) |
| MobileLLM | 125M; 350M | 2024.2 | Pre-train | Generic| [Github](https://github.com/facebookresearch/MobileLLM)  | [HF](https://huggingface.co/collections/facebook/mobilellm-6722be18cb86c20ebe113e95) | [Paper](https://arxiv.org/abs/2402.14905) |
| StableLM | 3B; 7B | 2023.4 | Pre-train | Generic| [Github](https://github.com/Stability-AI/StableLM) | [HF](https://huggingface.co/stabilityai/stablelm-3b-4e1t) | [Paper](https://huggingface.co/stabilityai/stablelm-3b-4e1t) |
| StableLM 2 | 1.6B | 2024.2 | Pre-train | Generic | [Github](https://github.com/Stability-AI/StableLM) | [HF](https://huggingface.co/stabilityai/stablelm-2-1_6b) | [Paper](https://arxiv.org/abs/2402.17834) |
| Cerebras-GPT | 111M-13B | 2023.4 | Pre-train | Generic | | [HF](https://huggingface.co/collections/cerebras/cerebras-gpt-66c623297a2370b8e670e0a1) | [Paper](https://arxiv.org/abs/2304.03208) |
| BLOOM, BLOOMZ | 560M; 1.1B; 1.7B; 3B; 7.1B; 176B | 2022.11 | Pre-train | Generic | | [HF](https://huggingface.co/bigscience) | [Paper](https://arxiv.org/abs/2211.05100) |
| OPT | 125M; 350M; 1.3B; 2.7B; 5.7B | 2022.5 | Pre-train | Generic | | [HF](https://huggingface.co/facebook/opt-350m) | [Paper](https://arxiv.org/abs/2205.01068) |
| XGLM | 1.7B; 2.9B; 7.5B | 2021.12 | Pre-train | Generic| [Github](https://github.com/facebookresearch/fairseq/tree/main/examples/xglm)  | [HF](https://huggingface.co/facebook/xglm-564M) | [Paper](https://aclanthology.org/2022.emnlp-main.616) |
| GPT-Neo | 125M; 350M; 1.3B; 2.7B | 2021.5 | Pre-train | Generic  | [Github](https://github.com/EleutherAI/gpt-neo/tree/master) |  | [Paper](https://zenodo.org/records/5297715) |
| Megatron-gpt2 | 355M; 2.5B; 8.3B | 2019.9  | Pre-train | Generic| [Github](https://github.com/NVIDIA/Megatron-LM)  |  | [Paper](https://arxiv.org/abs/1909.08053), [Blog](https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm) |
| MINITRON | 4B; 8B; 15B | 2024.7 | Pruning and Distillation | Generic | [Github](https://github.com/NVlabs/Minitron)  | [HF](https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base)| [Paper](https://arxiv.org/abs/2407.14679) |
| Orca 2 | 7B | 2023.11 | Distillation | Generic | | [HF](https://huggingface.co/microsoft/Orca-2-7b) |[Paper](https://arxiv.org/abs/2311.11045) |
| Dolly-v2 | 3B; 7B; 12B | 2023.4 | Instruction tuning | Generic| [Github](https://github.com/databrickslabs/dolly#getting-started-with-response-generation) | [HF](https://huggingface.co/databricks/dolly-v1-6b) | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| LaMini-LM | 61M-7B | 2023.4 | Distillation | Generic| [Github](https://github.com/mbzuai-nlp/LaMini-LM) | [HF](https://huggingface.co/databricks/dolly-v1-6b) | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| Specialized FlanT5 | 250M; 760M; 3B | 2023.1 | Instruction Tuning | Generic (math) | [Github](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)  | - | [Paper](https://proceedings.mlr.press/v202/fu23d.html) |
| FlanT5 | 80M; 250M; 780M; 3B | 2022.10 | Instruction Tuning | Generic | [Gihub](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) | [HF](https://huggingface.co/google/flan-t5-xxl) | [Paper](https://arxiv.org/abs/2210.11416) |
| T5 | 60M; 220M; 770M; 3B; 11B | 2019.9 | Pre-train | Generic | [Github](https://github.com/google-research/text-to-text-transfer-transformer)   | [HF](https://huggingface.co/google/t5-v1_1-base) | [Paper](https://arxiv.org/abs/1910.10683) |

## Table of Contents

- [SLM Survey](#slm-survey)
  - [Table of Contents](#table-of-contents)
  - [Overview of SLMs](#overview-of-slms)
  - [Timeline of SLMs](#timeline-of-slms)
  - [SLMs Paper List](#slms-paper-list)
    - [Existing SLMs](#existing-slms)
    - [Foundational Concepts in Building Language Models](#foundational-concepts-in-building-language-models)
    - [Advanced enhancement methods for SLM](#advanced-enhancement-methods-for-slm)
      - [Training from scratch](#training-from-scratch)
      - [Supervised fine-tuning](#supervised-fine-tuning)
      - [Data quality in KD](#data-quality-in-kd)
      - [Distillation for SLM](#distillation-for-slm)
      - [Quantization](#quantization)
      - [LLMs for SLM](#llms-for-slm)
    - [Task-specific SLM Applications](#task-specific-slm-applications)
      - [SLM in QA](#slm-in-qa)
      - [SLM in Coding](#slm-in-coding)
      - [SLM in Recommendation](#slm-in-recommendation)
      - [SLM in Web Search](#slm-in-web-search)
      - [SLM in Mobile-device](#slm-in-mobile-device)
    - [On-device Deployment Optimization Techniques](#on-device-deployment-optimization-techniques) 
      - [Memory Efficiency Optimization](#memory-efficiency-optimization)
      - [Runtime Efficiency Optimization](#runtime-efficiency-optimization)

### SLMs enhance LLMs
#### SLMs for LLMs Calibration

1. **Calibrating Large Language Models Using Their Generations Only.** *Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, Seong Joon Oh*. ACL 2024 Long, [[pdf]](https://aclanthology.org/2024.acl-long.824/) [[code]](https://github.com/parameterlab/apricot)
2. **Pareto Optimal Learning for Estimating Large Language Model Errors.** *Theodore Zhao, Mu Wei, J. Samuel Preston, Hoifung Poon*. ACL 2024 Long, [[pdf]](https://aclanthology.org/2024.acl-long.566/)
3. **The Internal State of an LLM Knows When It’s Lying.** *Amos Azaria, Tom Mitchell*. EMNLP 2023 Findings. [[pdf]](https://aclanthology.org/2023.findings-emnlp.68/)
#### SLMs for LLMs RAG
1. **Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs.** *Jiejun Tan, Zhicheng Dou, Yutao Zhu, Peidong Guo, Kun Fang, Ji-Rong Wen.* ACL 2024 Long.  [[pdf]](https://aclanthology.org/2024.acl-long.242/) [[code]](https://github.com/plageon/SlimPlm) [[huggingface]](https://huggingface.co/zstanjj/SlimPLM-Query-Rewriting)
2. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** *Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi.* ICLR 2024 Oral. [[pdf]](https://openreview.net/forum?id=hSyW5go0v8) [[huggingface]](https://huggingface.co/papers/2310.11511) [[code]](https://github.com/AkariAsai/self-rag) [[website]](https://selfrag.github.io/) [[model]](https://huggingface.co/selfrag/selfrag_llama2_7b) [[data]](https://huggingface.co/datasets/selfrag/selfrag_train_data) 
3. **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.** *Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu.* ICLR 2024 Workshop ME-FoMo Poster. [[pdf]](https://openreview.net/forum?id=9YvfRrpmyw) 
4. **Corrective Retrieval Augmented Generation.** *Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling.* arXiv 2024.1. [[pdf]](https://arxiv.org/abs/2401.15884) [[code]](https://github.com/HuskyInSalt/CRAG)
5. **Self-Knowledge Guided Retrieval Augmentation for Large Language Models.** *Yile Wang, Peng Li, Maosong Sun, Yang Liu.* EMNLP 2023 Findings. [[pdf]](https://aclanthology.org/2023.findings-emnlp.691/) [[code]](https://github.com/THUNLP-MT/SKR)
6.  **In-Context Retrieval-Augmented Language Models.** *Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham.* TACL 2023. [[pdf]](https://aclanthology.org/2023.tacl-1.75/) [[code]](https://github.com/AI21Labs/in-context-ralm)

### Foundational Concepts in Building Language Models

1. <u>SqueezeLLM</u>: **"SqueezeLLM: Dense-and-Sparse Quantization"**. *Sehoon Kim et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2306.07629)] [[Github](https://github.com/SqueezeAILab/SqueezeLLM)]
2. <u>JSQ</u>: **"Compressing Large Language Models by Joint Sparsification and Quantization"**. *Jinyang Guo et al.* PMLR 2024. [[Paper](https://proceedings.mlr.press/v235/guo24g.html)] [[Github](https://github.com/uanu2002/JSQ)]
3. <u>FrameQuant</u>: **"FrameQuant: Flexible Low-Bit Quantization for Transformers"**. *Harshavardhan Adepu et al.* 2024. [[Paper](https://arxiv.org/abs/2403.06082)] [[Github](https://github.com/vsingh-group/FrameQuant)]
4. <u>OneBit</u>: **"OneBit: Towards Extremely Low-bit Large Language Models"**. *Yuzhuang Xu et al.* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.11295)]
5. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang et al.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
6. <u>LQER</u>: **"LQER: Low-Rank Quantization Error Reconstruction for LLMs"**. *Cheng Zhang et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.02446)] [[Github](https://github.com/ChengZhang-98/lqer)]
7. <u>I-LLM</u>: **"I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models"**. *Xing Hu et al.* 2024. [[Paper](https://arxiv.org/abs/2405.17849)] [[Github](https://anonymous.4open.science/r/I-LLM-F242/README.md)]
8. <u>PV-Tuning</u>: **"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"**. *Vladimir Malinovskii et al.* 2024. [[Paper](https://arxiv.org/abs/2405.14852)]
9. <u>BitNet</u>: **"BitNet: Scaling 1-bit Transformers for Large Language Models"**. *Hongyu Wang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.11453)]
10. <u>BitNet b1.58</u>: **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**. *Shuming Ma et al.* 2024. [[Paper](https://arxiv.org/abs/2402.17764)]
11. <u>PEQA</u>: **"Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization"**. *Jeonghoon Kim et al.* NIPS 2023. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667691)]
12. <u>QLoRA</u>: **"QLORA: efficient finetuning of quantized LLMs"**. *Tim Dettmers et al.* NIPS 2023. [[Paper](https://dl.acm.org/doi/abs/10.5555/3666122.3666563)] [[Github](https://github.com/artidoro/qlora)]
  

### Advanced enhancement methods for SLM

#### Training from scratch

1. <u>MindLLM</u>: **"MindLLM: Pre-training Lightweight Large Language Model from Scratch, Evaluations and Domain Applications"**. *Yizhe Yang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.15777)] [[HuggingFace](https://huggingface.co/bit-dny/MindLLM-1b3-chat-zh-v2.0)]
2. <u>MobiLlama</u>: **"MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT"**. *Omkar Thawakar et al.* 2024. [[Paper](https://arxiv.org/abs/2402.16840)] [[Github](https://github.com/mbzuai-oryx/MobiLlama)] [[HuggingFace](https://huggingface.co/collections/MBZUAI/mobillama-65dd4182d588c91e8230332e)]
3. <u>MobileLLM</u>: **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"**. *Zechun Liu et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.14905)] [[Github](https://github.com/facebookresearch/MobileLLM)] [[HuggingFace](https://huggingface.co/papers/2402.14905)]


#### Supervised fine-tuning

1. <u>MobileBERT</u>: **"MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"**. *Zhiqing Sun et al.* ACL 2020. [[Paper](https://arxiv.org/abs/2004.02984)] [[Github](https://github.com/google-research/google-research/tree/master/mobilebert)] [[HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/mobilebert)]
2. <u>Alpaca 7B</u>: **"Alpaca: A Strong, Replicable Instruction-Following Model"**. *Rohan Taori et al.* 2023. [[Paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)]
3. <u>RLHF</u>: **"Training language models to follow instructions with human feedback"**. *Long Ouyang et al.* 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
4. <u>DPO</u>: **"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"**. *Rafael Rafailov et al.* 2024. [[Paper](https://arxiv.org/abs/2305.18290)]

#### Data quality in KD

1. <u>TinyStory</u>: **"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"**. *Ronen Eldan et al.* 2023. [[Paper](https://arxiv.org/abs/2305.07759)] [[HuggingFace](https://huggingface.co/papers/2305.07759)]
2. <u>AS-ES</u>: **"AS-ES Learning: Towards Efficient CoT Learning in Small Models"**. *Nuwa Xi et al.* 2024. [[Paper](https://arxiv.org/abs/2403.01969)]
3. <u>Self-Amplify</u>: **"Self-AMPLIFY: Improving Small Language Models with Self Post Hoc Explanations"**. *Milan Bhan et al.* 2024. [[Paper](https://arxiv.org/abs/2402.12038)] 

#### Distillation for SLM

1. <u>GKD</u>: **"On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"**. *Rishabh Agarwal et al.* ICLR 2024. [[Paper](https://arxiv.org/abs/2306.13649)] 
2. <u>DistilLLM</u>: **"DistiLLM: Towards Streamlined Distillation for Large Language Models"**. *Jongwoo Ko et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.03898)] [[Github](https://github.com/jongwooko/distillm)]
3. <u>Adapt-and-Distill</u>: **"Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains"**. *Yunzhi Yao et al.* ACL2021. [[Paper](https://arxiv.org/abs/2106.13474)] [[Github](https://github.com/microsoft/unilm/tree/master/adalm)] 

#### Quantization

1. <u>SmoothQuant</u>: **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**. *Guangxuan Xiao et al.* ICML 2023. [[Paper](https://arxiv.org/abs/2211.10438)] [[Github](https://github.com/mit-han-lab/smoothquant)]
2. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang et al.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
3. <u>LLM-QAT</u>: **"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"**. *Zechun Liu et al.* 2023. [[Paper](https://arxiv.org/abs/2305.17888)]
4. <u>PB-LLM</u>: **"PB-LLM: Partially Binarized Large Language Models"**. *Yuzhang Shang et al.* 202X. [[Paper](https://arxiv.org/abs/2310.00034)] [[Github](https://github.com/hahnyuan/PB-LLM)]
5. <u>OneBit</u>: **"OneBit: Towards Extremely Low-bit Large Language Models"**. *Yuzhuang Xu et al.* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.11295)]
6. <u>BitNet</u>: **"BitNet: Scaling 1-bit Transformers for Large Language Models"**. *Hongyu Wang et al.* 2023. [[Paper](https://arxiv.org/abs/2310.11453)]
7. <u>BitNet b1.58</u>: **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**. *Shuming Ma et al.* 2024. [[Paper](https://arxiv.org/abs/2402.17764)]

#### LLMs for SLM

1. <u>Ma et al.</u>: **"Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!"**. *Yubo Ma et al.* EMNLP 2023. [[Paper](https://arxiv.org/abs/2303.08559)] [[Github](https://github.com/mayubo2333/LLM-IE)]
2. <u>MoQE</u>: **"Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness"**. *Young Jin Kim et al.* 2023. [[Paper](https://arxiv.org/abs/2310.02410)]
3. <u>SLM-RAG</u>: **"Can Small Language Models With Retrieval-Augmented Generation Replace Large Language Models When Learning Computer Science?"**. *Suqing Liu et al.* ITiCSE 2024. [[Paper](https://dl.acm.org/doi/10.1145/3649217.3653554)] 

### Task-specific SLM Applications
#### SLM in QA

1. <u>Alpaca</u>: **"Alpaca: A Strong, Replicable Instruction-Following Model"**. *Rohan Taori et al.*  202X. [[Paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/stabilityai/StableBeluga2)] 
2. <u>Stable Beluga 7B</u>: **"Stable Beluga 2"**. *Mahan et al.*  2023. [[HuggingFace](https://huggingface.co/stabilityai/StableBeluga2)] 
3. <u>Fine-tuned BioGPT Guo et al.</u>: **"Improving Small Language Models on PubMedQA via Generative Data Augmentation"**. *Zhen Guo et al.*  2023. [[Paper](https://arxiv.org/abs/2305.07804)]
4. <u>Financial SLMs</u>: **"Fine-tuning Smaller Language Models for Question Answering over Financial Documents"**. *Karmvir Singh Phogat et al.*  2024. [[Paper](https://arxiv.org/abs/2408.12337)]
5. <u>ColBERT</u>: **"ColBERT Retrieval and Ensemble Response Scoring for Language Model Question Answering"**. *Alex Gichamba et al.*  IEEE 2024. [[Paper](https://arxiv.org/abs/2408.10808)] 
6. <u>T-SAS</u>: **"Test-Time Self-Adaptive Small Language Models for Question Answering"**. *Soyeong Jeong et al.* ACL 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.1033)] [[Github](https://github.com/starsuzi/T-SAS)]
7. <u>Rationale Ranking</u>: **"Answering Unseen Questions With Smaller Language Models Using Rationale Generation and Dense Retrieval"**. *Tim Hartill et al.*  2023. [[Paper](https://arxiv.org/abs/2308.04711)]

#### SLM in Coding

1. <u>Phi-3.5-mini</u>: **"Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"**. *Marah Abdin et al.*  202X. [[Paper](https://arxiv.org/abs/2404.14219)] [[HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)] 
2. <u>TinyLlama</u>: **"TinyLlama: An Open-Source Small Language Model"**. *Peiyuan Zhang et al.*  2024. [[Paper](https://arxiv.org/abs/2401.02385)] [[HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)] 
3. <u>CodeLlama</u>: **"Code Llama: Open Foundation Models for Code"**. *Baptiste Rozière et al.*  2024. [[Paper](https://arxiv.org/abs/2308.12950)] [[HuggingFace](https://huggingface.co/codellama)] 
4. <u>CodeGemma</u>: **"CodeGemma: Open Code Models Based on Gemma"**. *Heri Zhao et al.*  2024. [[Paper](https://arxiv.org/abs/2406.11409)] [[HuggingFace](https://huggingface.co/google/codegemma-7b-it)] 

#### SLM in Recommendation

1. <u>PromptRec</u>: **"Could Small Language Models Serve as Recommenders? Towards Data-centric Cold-start Recommendations"**. *Xuansheng Wu, et al.*  2024. [[Paper](https://arxiv.org/abs/2306.17256v4)] [[Github](https://github.com/JacksonWuxs/PromptRec)] 
2. <u>SLIM</u>: **"Can Small Language Models be Good Reasoners for Sequential Recommendation?"**. *Yuling Wang et al.*  2024. [[Paper](https://arxiv.org/abs/2403.04260v1)]
3. <u>BiLLP</u>: **"Large Language Models are Learnable Planners for Long-Term Recommendation"**. *Wentao Shi et al.*  2024. [[Paper](https://arxiv.org/abs/2403.00843v2)] 
4. <u>ONCE</u>: **"ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models"**. *Qijiong Liu et al.*  2023. [[Paper](https://arxiv.org/abs/2305.06566)] [[Github]()] [[HuggingFace]()] 
5. <u>RecLoRA</u>: **"Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation"**. *Jiachen Zhu et al.*  2024. [[Paper](https://arxiv.org/abs/2408.03533v2)] 

#### SLM in Web Search

1. <u>Content encoder</u>: **"Pre-training Tasks for Embedding-based Large-scale Retrieval"**. *Wei-Cheng Chang et al.*  ICLR 2020. [[Paper](https://arxiv.org/abs/2002.03932)]
2. <u>Poly-encoders</u>: **"Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring"**. *Samuel Humeau et al.*  ICLR 2020. [[Paper](https://arxiv.org/abs/1905.01969)]
3. <u>Twin-BERT</u>: **"TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval"**. *Wenhao Lu et al.*  2020. [[Paper](https://arxiv.org/abs/2002.06275)]
4. <u>H-ERNIE</u>: **"H-ERNIE: A Multi-Granularity Pre-Trained Language Model for Web Search"**. *Xiaokai Chu et al.*  SIGIR 2022. [[Paper](https://dl.acm.org/doi/10.1145/3477495.3531986)]
5. <u>Ranker</u>: **"Passage Re-ranking with BERT"**. *Rodrigo Nogueira et al.*  2019. [[Paper](https://arxiv.org/abs/1901.04085)] [[Github](https://github.com/nyu-dl/dl4marco-bert)]
6. <u>Rewriter</u>: **"Query Rewriting for Retrieval-Augmented Large Language Models"**. *Xinbei Ma et al.*  EMNLP2023. [[Paper](https://arxiv.org/abs/2305.14283)] [[Github](https://github.com/xbmxb/RAG-query-rewriting)]

#### SLM in Mobile-device

1. <u>Octopus</u>: **"Octopus: On-device language model for function calling of software APIs"**. *Wei Chen et al.*  2024. [[Paper](https://arxiv.org/abs/2404.01549)]
2. <u>MobileAgent</u>: **"Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration"**. *Junyang Wang et al.*  2024. [[Paper](https://arxiv.org/abs/2406.01014)] [[Github](https://github.com/X-PLUG/MobileAgent)] [[HuggingFace](https://huggingface.co/spaces/junyangwang0410/Mobile-Agent)]
3. <u>Revolutionizing Mobile Interaction</u>: **"Revolutionizing Mobile Interaction: Enabling a 3 Billion Parameter GPT LLM on Mobile"**. *Samuel Carreira et al.*  2023. [[Paper](https://arxiv.org/abs/2310.01434)] 
4. <u>AutoDroid</u>: **"AutoDroid: LLM-powered Task Automation in Android"**. *Hao Wen et al.*  2023. [[Paper](https://arxiv.org/abs/2308.15272)]
5. <u>On-device Agent for Text Rewriting</u>: **"Towards an On-device Agent for Text Rewriting"**. *Yun Zhu et al.*  2023. [[Paper](https://arxiv.org/abs/2308.11807)]

### On-device Deployment Optimization Techniques
#### Memory Efficiency Optimization

1. <u>EDGE-LLM</u>: **"EDGE-LLM: Enabling Efficient Large Language Model Adaptation on Edge Devices via Layerwise Unified Compression and Adaptive Layer Tuning and Voting"**. *Zhongzhi Yu et al.*  2024. [[Paper](https://arxiv.org/abs/2406.15758)] [[Github](https://github.com/GATECH-EIC/Edge-LLM)]
2. <u>LLM-PQ</u>: **"LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization"**. *Juntao Zhao et al.*  2024. [[Paper](https://arxiv.org/abs/2403.01136)] [[Github](https://github.com/tonyzhao-jt/LLM-PQ?tab=readme-ov-file)]
3. <u>AWQ</u>: **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"**. *Ji Lin et al.*  MLSys 2024. [[Paper](https://arxiv.org/abs/2306.00978)] [[Github](https://github.com/mit-han-lab/llm-awq)]
4. <u>MobileAIBench</u>: **"MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases"**. *Rithesh Murthy et al.*  2024. [[Paper](https://arxiv.org/abs/2406.10290)] [[Github](https://github.com/XiaoMi/mobile-ai-bench)]
5. <u>MobileLLM</u>: **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"**. *Zechun Liu et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.14905)] [[Github](https://github.com/facebookresearch/MobileLLM)] [[HuggingFace](https://huggingface.co/papers/2402.14905)]
6. <u>EdgeMoE</u>: **"EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models"**. *Rongjie Yi et al.*  2023. [[Paper](https://arxiv.org/abs/2308.14352)] [[Github](https://github.com/sharc-lab/Edge-MoE)] 
7. <u>GEAR</u>: **"GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM"**. *Hao Kang et al.*  2024. [[Paper](https://arxiv.org/abs/2403.05527)] [[Github](https://github.com/opengear-project/GEAR)]
8. <u>DMC</u>: **"Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference"**. *Name et al.*  202X. [[Paper](https://arxiv.org/abs/2403.09636)]
9. <u>Transformer-Lite</u>: **"Transformer-Lite: High-efficiency Deployment of Large Language Models on Mobile Phone GPUs"**. *Luchang Li et al.*  202X. [[Paper](https://arxiv.org/abs/2403.20041)]
10. <u>LLMaaS</u>: **"Wangsong Yin"**. *Piotr Nawrot et al.*  2024. [[Paper](https://arxiv.org/abs/2403.11805)]

#### Runtime Efficiency Optimization

1. <u>EdgeMoE</u>: **"EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models"**. *Rongjie Yi et al.*  2023. [[Paper](https://arxiv.org/abs/2308.14352)] [[Github](https://github.com/sharc-lab/Edge-MoE)] 
2. <u>LLMCad</u>: **"LLMCad: Fast and Scalable On-device Large Language Model Inference"**. *Daliang Xu et al.*  2023. [[Paper](https://arxiv.org/abs/2309.04255)]
3. <u>LinguaLinked</u>: **"LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices"**. *Junchen Zhao et al.*  2023 [[Paper](https://arxiv.org/abs/2312.00388)]



<!-- Insertion Template: 0. <u>Model</u>: **"Title"**. *Name et al.*  202X. [[Paper]()] [[Github]()] [[HuggingFace]()] -->

