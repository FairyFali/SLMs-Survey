# SLMs Survey

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![](https://img.shields.io/badge/PaperNumber-133-brightgreen) ![](https://img.shields.io/badge/PRs-Welcome-red)


## A Comprehensive Survey of Small Language Models: Technology, On-Device Applications, Efficiency, Enhancements for LLMs, and Trustworthiness
This repo includes the papers discussed in our latest survey paper on small language models.    
:book: Read the full paper here: [Paper Link](https://arxiv.org/abs/2411.03350)

## News
* 2024/12/28: The second version of our survey is on Arxiv!
* 2024/11/04: The first version of our survey is on Arxiv!

## Reference
If our survey is useful for your research, please kindly cite our [paper](https://arxiv.org/abs/2411.03350):
```
@article{wang2024comprehensive,
  title={A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness},
  author={Wang, Fali and Zhang, Zhiwei and Zhang, Xianren and Wu, Zongyu and Mo, Tzuhao and Lu, Qiuhao and Wang, Wanjing and Li, Rui and Xu, Junjie and Tang, Xianfeng and others},
  journal={arXiv preprint arXiv:2411.03350},
  year={2024}
}
```

## Overview of SLMs
![Overview of Small Language Models](images/overview_structure.png)

## Timeline of SLMs
![Timeline of Small Language Models](images/overview_of_small_language_models.PNG)

## SLMs Paper List
### Existing SLMs


| Model | #Params | Date | Paradigm | Domain | Code | HF Model | Paper/Blog |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |---- |
| PhoneLM | 0.5B; 1.5B | 2024.11 | Pre-train | Generic | [Github](https://github.com/UbiquitousLearning/PhoneLM)  | [HF](https://huggingface.co/mllmTeam/PhoneLM-0.5B) | [Paper](https://arxiv.org/abs/2411.05046) |
| Llama 3.2 | 1B; 3B | 2024.9 | Pre-train | Generic | [Github](https://github.com/meta-llama/llama-models)  | [HF](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf) | [Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) |
| Qwen 1 | 1.8B; 7B; 14B; 72B | 2023.12 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2309.16609) |
| Qwen 1.5 | 0.5B; 1.8B; 4B; 7B; 14B; 32B; 72B | 2024.2 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2309.16609) |
| Qwen 2 | 0.5B; 1.5B; 7B; 57B; 72B | 2024.6 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2407.10671) |
| Qwen 2.5 | 0.5B; 1.5B; 3B; 7B; 14B; 32B; 72B | 2024.9 | Pre-train | Generic | [Github](https://github.com/QwenLM/Qwen) | [HF](https://huggingface.co/Qwen) | [Paper](https://arxiv.org/abs/2407.10671) |
| Gemma | 2B; 7B | 2024.2 | Pre-train | Generic | | [HF](https://huggingface.co/collections/google/gemma-release-65d5efbccdbb8c4202ec078b) | [Paper](https://arxiv.org/abs/2403.08295) |
| Gemma 2 | 2B; 9B; 27B | 2024.7 | Pre-train | Generic | | [HF](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315) | [Paper](https://arxiv.org/abs/2408.00118) |
| SmolLM | 135M; 360M; 1.7B | 2024.7 | Pre-train | Generic | [Github](https://github.com/huggingface/smollm) | [HF](https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966) | [Blog](https://huggingface.co/blog/smollm) |
| H2O-Danube3 | 500M; 4B | 2024.7 | Pre-train | Generic | | [HF](https://huggingface.co/collections/h2oai/h2o-danube3-6687a993641452457854c609) | [Paper](https://arxiv.org/abs/2407.09276) |
| LLM-Neo | 1B | 2024.11 | Continous Training | Generic | | [HF](https://huggingface.co/yang31210999/Llama-3.2-1B-Instruct-Neo-BAAI-10k) | [Paper](https://arxiv.org/pdf/2411.06839) |
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
| Galactica | 125M; 1.3B; 6.7B | 2022.11 | Pre-train | Scientific | | [HF](https://huggingface.co/NuclearnAI/SPARK-mini-base) | [Paper](https://arxiv.org/abs/2211.09085) |
| OPT | 125M; 350M; 1.3B; 2.7B; 5.7B | 2022.5 | Pre-train | Generic | | [HF](https://huggingface.co/facebook/opt-350m) | [Paper](https://arxiv.org/abs/2205.01068) |
| XGLM | 1.7B; 2.9B; 7.5B | 2021.12 | Pre-train | Generic| [Github](https://github.com/facebookresearch/fairseq/tree/main/examples/xglm)  | [HF](https://huggingface.co/facebook/xglm-564M) | [Paper](https://aclanthology.org/2022.emnlp-main.616) |
| GPT-Neo | 125M; 350M; 1.3B; 2.7B | 2021.5 | Pre-train | Generic  | [Github](https://github.com/EleutherAI/gpt-neo/tree/master) |  | [Paper](https://zenodo.org/records/5297715) |
| Megatron-gpt2 | 355M; 2.5B; 8.3B | 2019.9  | Pre-train | Generic| [Github](https://github.com/NVIDIA/Megatron-LM)  |  | [Paper](https://arxiv.org/abs/1909.08053), [Blog](https://huggingface.co/docs/accelerate/en/usage_guides/megatron_lm) |
| MINITRON | 4B; 8B; 15B | 2024.7 | Pruning and Distillation | Generic | [Github](https://github.com/NVlabs/Minitron)  | [HF](https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Width-Base)| [Paper](https://arxiv.org/abs/2407.14679) |
| MiniMix | 7B | 2024.7 | Pre-train | Generic | [Github](https://github.com/GeneZC/MiniMA)  | [HF](https://huggingface.co/GeneZC/MiniMix-2_4x3B)| [Paper](https://arxiv.org/abs/2311.07052) |
| MiniMA-2 | 1B; 3B | 2023.12 | Pre-train | Generic | [Github](https://github.com/GeneZC/MiniMA)  | [HF](https://huggingface.co/GeneZC/MiniMA-2-3B)| [Paper](https://arxiv.org/abs/2311.07052) |
| MiniMA | 3B | 2023.11 | Pruning and Distillation | Generic | [Github](https://github.com/GeneZC/MiniMA)  | [HF](https://huggingface.co/GeneZC/MiniMA-3B)| [Paper](https://arxiv.org/abs/2311.07052) |
| Orca 2 | 7B | 2023.11 | Distillation | Generic | | [HF](https://huggingface.co/microsoft/Orca-2-7b) |[Paper](https://arxiv.org/abs/2311.11045) |
| Dolly-v2 | 3B; 7B; 12B | 2023.4 | Instruction tuning | Generic| [Github](https://github.com/databrickslabs/dolly#getting-started-with-response-generation) | [HF](https://huggingface.co/databricks/dolly-v1-6b) | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| LaMini-LM | 61M-7B | 2023.4 | Distillation | Generic| [Github](https://github.com/mbzuai-nlp/LaMini-LM) | [HF](https://huggingface.co/databricks/dolly-v1-6b) | [Blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) |
| Specialized FlanT5 | 250M; 760M; 3B | 2023.1 | Instruction Tuning | Generic (math) | [Github](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)  | - | [Paper](https://proceedings.mlr.press/v202/fu23d.html) |
| FlanT5 | 80M; 250M; 780M; 3B | 2022.10 | Instruction Tuning | Generic | [Gihub](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints) | [HF](https://huggingface.co/google/flan-t5-xxl) | [Paper](https://arxiv.org/abs/2210.11416) |
| T5 | 60M; 220M; 770M; 3B; 11B | 2019.9 | Pre-train | Generic | [Github](https://github.com/google-research/text-to-text-transfer-transformer)   | [HF](https://huggingface.co/google/t5-v1_1-base) | [Paper](https://arxiv.org/abs/1910.10683) |


<!--
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
-->
### SLM Architecture

1. Transformer: **Attention is all you need.** *Ashish Vaswani*. NeurIPS 2017. 
2. Mamba 1: **Mamba: Linear-time sequence modeling with selective state spaces.** *Albert Gu and Tri Dao*. COLM 2024.  [[Paper]](https://openreview.net/forum?id=tEYskw1VY2#discussion).
3. Mamba 2: **Transformers are SSMs: Generalized models and efficient algorithms through structured state space duality.** *Tri Dao and Albert Gu*. ICML 2024. [[Paper]](https://openreview.net/forum?id=ztn8FCR1td) [[Code]](https://github.com/state-spaces/mamba)
4. **Hymba: A Hybrid-head Architecture for Small Language Models.** *Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, Yingyan Lin, Jan Kautz, Pavlo Molchanov*. arXiv 2024.11. [[Paper]](https://arxiv.org/abs/2411.13676) [[HF]](https://huggingface.co/nvidia/Hymba-1.5B-Base)
5. **xLSTM: Extended Long Short-Term Memory.** *Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, Sepp Hochreiter*. arXiv 2024.12. [[Paper]](https://arxiv.org/abs/2405.04517) [[Code]](https://github.com/NX-AI/xlstm)




### Enhancement for SLM

#### Training from scratch

1. <u>MobiLlama</u>: **"MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT"**. *Omkar Thawakar, Ashmal Vayani, Salman Khan, Hisham Cholakal, Rao M. Anwer, Michael Felsberg, Tim Baldwin, Eric P. Xing, Fahad Shahbaz Khan.* arXiv 2024. [[Paper](https://arxiv.org/abs/2402.16840)] [[Github](https://github.com/mbzuai-oryx/MobiLlama)] [[HuggingFace](https://huggingface.co/collections/MBZUAI/mobillama-65dd4182d588c91e8230332e)]
2. <u>MobileLLM</u>: **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"**. *Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra* ICML 2024. [[Paper](https://arxiv.org/abs/2402.14905)] [[Github](https://github.com/facebookresearch/MobileLLM)] [[HuggingFace](https://huggingface.co/papers/2402.14905)]
3. **Rethinking optimization and architecture for tiny language models.** *Yehui Tang, Fangcheng Liu, Yunsheng Ni, Yuchuan Tian, Zheyuan Bai, Yi-Qi Hu, Sichao Liu, Shangling Jui, Kai Han, and Yunhe Wang.* ICML 2024. [[Paper]](https://openreview.net/forum?id=mHIEOZtDDF) [[Code]](https://github.com/YuchuanTian/RethinkTinyLM)
4. <u>MindLLM</u>: **"MindLLM: Pre-training Lightweight Large Language Model from Scratch, Evaluations and Domain Applications"**. *Yizhe Yang, Huashan Sun, Jiawei Li, Runheng Liu, Yinghao Li, Yuhang Liu, Heyan Huang, Yang Gao*. arXiv 2023. [[Paper](https://arxiv.org/abs/2310.15777)] [[HuggingFace](https://huggingface.co/bit-dny/MindLLM-1b3-chat-zh-v2.0)]


#### Supervised fine-tuning

1. **Direct preference optimization: Your language model is secretly a reward model.** *Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn.* NeurIPS, 2024. [[Paper](https://arxiv.org/abs/2305.18290)] [[Code]](https://github.com/eric-mitchell/direct-preference-optimization)
2. **Enhancing chat language models by scaling high-quality instructional conversations.** *Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou.* EMNLP 2023. [[Paper]](https://aclanthology.org/2023.emnlp-main.183/) [[Code]](https://github.com/thunlp/UltraChat)
3. **SlimOrca: An Open Dataset of GPT-4 Augmented FLAN Reasoning Traces, with Verification.** *Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium".* Huggingface, 2023. [[Data]](https://huggingface.co/datasets/Open-Orca/SlimOrca)
4. **Stanford Alpaca: An Instruction-following LLaMA model.** *Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto.* GitHub, 2023. [[Blog](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/tatsu-lab/alpaca-7b-wdiff)]
5. **OpenChat: Advancing Open-source Language Models with Mixed-Quality Data.** *Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu.* ICLR, 2024. [[Paper]](https://openreview.net/forum?id=AOJyfhWYHf) [[Code]](https://github.com/imoneoi/openchat) [[HuggingFace]](https://huggingface.co/openchat)
6. **Training language models to follow instructions with human feedback.** *Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, Ryan Lowe.* NeurIPS, 2022. [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)
7. <u>RLHF</u>: **"Training language models to follow instructions with human feedback"**. *Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe.* 2022. [[Paper](https://arxiv.org/abs/2203.02155)]
8. <u>MobileBERT</u>: **"MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices"**. *Zhiqing Sun, Hongkun Yu, Xiaodan Song, Renjie Liu, Yiming Yang, Denny Zhou.* ACL 2020. [[Paper](https://arxiv.org/abs/2004.02984)] [[Github](https://github.com/google-research/google-research/tree/master/mobilebert)] [[HuggingFace](https://huggingface.co/docs/transformers/en/model_doc/mobilebert)]
9. **Language models are unsupervised multitask learners.** *Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever.* OpenAI Blog, 2019. [[Paper]](https://insightcivic.s3.us-east-1.amazonaws.com/language-models.pdf)


#### Data quality in KD

1. <u>TinyStory</u>: **"TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"**. *Ronen Eldan, Yuanzhi Li.* 2023. [[Paper](https://arxiv.org/abs/2305.07759)] [[HuggingFace](https://huggingface.co/papers/2305.07759)]
2. <u>AS-ES</u>: **"AS-ES Learning: Towards Efficient CoT Learning in Small Models"**. *Nuwa Xi, Yuhan Chen, Sendong Zhao, Haochun Wang, Bing Qin, Ting Liu.* 2024. [[Paper](https://arxiv.org/abs/2403.01969)]
3. <u>Self-Amplify</u>: **"Self-AMPLIFY: Improving Small Language Models with Self Post Hoc Explanations"**. *Milan Bhan, Jean-Noel Vittaut, Nicolas Chesneau, Marie-Jeanne Lesot.* 2024. [[Paper](https://arxiv.org/abs/2402.12038)] 
4. **Large Language Models Can Self-Improve.** *Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han.* EMNLP 2023. [[Paper]](https://aclanthology.org/2023.emnlp-main.67/) 
5. **Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing.** *Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Haitao Mi, and Dong Yu.* NeurIPS 2024. [[Paper]](https://openreview.net/forum?id=tPdJ2qHkOB) [[Code]](https://github.com/YeTianJHU/AlphaLLM)

#### Distillation for SLMs


1. <u>GKD</u>: **"On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"**. *Rishabh Agarwal et al.* ICLR 2024. [[Paper](https://arxiv.org/abs/2306.13649)] 
2. <u>DistilLLM</u>: **"DistiLLM: Towards Streamlined Distillation for Large Language Models"**. *Jongwoo Ko et al.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.03898)] [[Github](https://github.com/jongwooko/distillm)]
3. <u>Adapt-and-Distill</u>: **"Adapt-and-Distill: Developing Small, Fast and Effective Pretrained Language Models for Domains"**. *Yunzhi Yao et al.* ACL2021. [[Paper](https://arxiv.org/abs/2106.13474)] [[Github](https://github.com/microsoft/unilm/tree/master/adalm)] 
4. <u>AKL</u>: **"Rethinking kullback-leibler divergence in knowledge distillation for large language models"**. *Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, Ngai Wong.* Arxiv 2024. [[Paper](https://arxiv.org/pdf/2404.02657)] [[Github](https://github.com/wutaiqiang/LLM_KD_AKL)]
5. **Weight-inherited distillation for task-agnostic bert compression** *Taiqiang Wu, Cheng Hou, Shanshan Lao, Jiayi Li, Ngai Wong, Zhe Zhao, Yujiu Yang* NAACL, 2024, [[Paper]](https://arxiv.org/abs/2305.09098) [[Code]](https://github.com/wutaiqiang/WID-NAACL2024)

#### Quantization

1. <u>SmoothQuant</u>: **"SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"**. *Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han.* ICML 2023. [[Paper](https://arxiv.org/abs/2211.10438)] [[Github](https://github.com/mit-han-lab/smoothquant)][[Slides](https://github.com/mit-han-lab/smoothquant/blob/main/assets/SmoothQuant.pdf)][[Video](https://www.youtube.com/watch?v=U0yvqjdMfr0)]
2. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
3. <u>LLM-QAT</u>: **"LLM-QAT: Data-Free Quantization Aware Training for Large Language Models"**. *Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, Vikas Chandra.* 2023. [[Paper](https://arxiv.org/abs/2305.17888)]
4. <u>PB-LLM</u>: **"PB-LLM: Partially Binarized Large Language Models"**. *Zhihang Yuan, Yuzhang Shang, Zhen Dong.* ICLR 2024. [[Paper](https://openreview.net/forum?id=BifeBRhikU)] [[Github](https://github.com/hahnyuan/PB-LLM)]
5. <u>OneBit</u>: **"OneBit: Towards Extremely Low-bit Large Language Models"**. *Yuzhuang Xu, Xu Han, Zonghan Yang, Shuo Wang, Qingfu Zhu, Zhiyuan Liu, Weidong Liu, Wanxiang Che.* NeurIPS 2024. [[Paper](https://arxiv.org/abs/2402.11295)]
6. <u>BitNet</u>: **"BitNet: Scaling 1-bit Transformers for Large Language Models"**. *Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei.* 2023. [[Paper](https://arxiv.org/abs/2310.11453)]
7. <u>BitNet b1.58</u>: **"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"**. *Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei.* 2024. [[Paper](https://arxiv.org/abs/2402.17764)]
8. <u>SqueezeLLM</u>: **"SqueezeLLM: Dense-and-Sparse Quantization"**. *Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, Kurt Keutzer.* ICML 2024. [[Paper](https://arxiv.org/abs/2306.07629)] [[Github](https://github.com/SqueezeAILab/SqueezeLLM)]
9. <u>JSQ</u>: **"Compressing Large Language Models by Joint Sparsification and Quantization"**. *Jinyang Guo, Jianyu Wu, Zining Wang, Jiaheng Liu, Ge Yang, Yifu Ding, Ruihao Gong, Haotong Qin, Xianglong Liu.* ICML 2024. [[Paper](https://proceedings.mlr.press/v235/guo24g.html)] [[Github](https://github.com/uanu2002/JSQ)]
10. <u>FrameQuant</u>: **"FrameQuant: Flexible Low-Bit Quantization for Transformers"**. *Harshavardhan Adepu, Zhanpeng Zeng, Li Zhang, Vikas Singh.* 2024. [[Paper](https://arxiv.org/abs/2403.06082)] [[Github](https://github.com/vsingh-group/FrameQuant)]
11. <u>BiLLM</u>: **"BiLLM: Pushing the Limit of Post-Training Quantization for LLMs"**. *Wei Huang, Yangdong Liu, Haotong Qin, Ying Li, Shiming Zhang, Xianglong Liu, Michele Magno, Xiaojuan Qi.* 2024. [[Paper](https://arxiv.org/abs/2402.04291)] [[Github](https://github.com/Aaronhuang-778/BiLLM)]
12. <u>LQER</u>: **"LQER: Low-Rank Quantization Error Reconstruction for LLMs"**. *Cheng Zhang, Jianyi Cheng, George A. Constantinides, Yiren Zhao.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.02446)] [[Github](https://github.com/ChengZhang-98/lqer)]
13. <u>I-LLM</u>: **"I-LLM: Efficient Integer-Only Inference for Fully-Quantized Low-Bit Large Language Models"**. *Xing Hu, Yuan Cheng, Dawei Yang, Zhihang Yuan, Jiangyong Yu, Chen Xu, Sifan Zhou.* 2024. [[Paper](https://arxiv.org/abs/2405.17849)] [[Github](https://anonymous.4open.science/r/I-LLM-F242/README.md)]
14. <u>PV-Tuning</u>: **"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"**. *Vladimir Malinovskii, Denis Mazur, Ivan Ilin, Denis Kuznedelev, Konstantin Burlachenko, Kai Yi, Dan Alistarh, Peter Richtarik.* 2024. [[Paper](https://arxiv.org/abs/2405.14852)]
15. <u>PEQA</u>: **"Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization"**. *Jeonghoon Kim, Jung Hyun Lee, Sungdong Kim, Joonsuk Park, Kang Min Yoo, Se Jung Kwon, Dongsoo Lee.* NIPS 2023. [[Paper](https://dl.acm.org/doi/10.5555/3666122.3667691)]
16. <u>QLoRA</u>: **"QLORA: efficient finetuning of quantized LLMs"**. *Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke ZettlemoyerAuthors Info & Claims.* NIPS 2023. [[Paper](https://dl.acm.org/doi/abs/10.5555/3666122.3666563)] [[Github](https://github.com/artidoro/qlora)]

#### LLM techniques for SLMs

1. **"Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!"**. *Yubo Ma, Yixin Cao, YongChing Hong, Aixin Sun.* EMNLP 2023. [[Paper](https://arxiv.org/abs/2303.08559)] [[Github](https://github.com/mayubo2333/LLM-IE)]
2. <u>MoQE</u>: **"Mixture of Quantized Experts (MoQE): Complementary Effect of Low-bit Quantization and Robustness"**. *Young Jin Kim, Raffy Fahim, Hany Hassan Awadalla.* 2023. [[Paper](https://arxiv.org/abs/2310.02410)]
3. <u>SLM-RAG</u>: **"Can Small Language Models With Retrieval-Augmented Generation Replace Large Language Models When Learning Computer Science?"**. *Suqing Liu, Zezhu Yu, Feiran Huang, Yousef Bulbulia, Andreas Bergen, Michael Liut.* ITiCSE 2024. [[Paper](https://dl.acm.org/doi/10.1145/3649217.3653554)] 

### Task-specific SLM Applications

#### SLM in QA

1. <u>Alpaca</u>: **"Alpaca: A Strong, Replicable Instruction-Following Model"**. *Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto.*  2023. [[Paper](https://crfm.stanford.edu/2023/03/13/alpaca.html)] [[Github](https://github.com/tatsu-lab/stanford_alpaca)] [[HuggingFace](https://huggingface.co/stabilityai/StableBeluga2)] [[Website](https://crfm.stanford.edu/2023/03/13/alpaca.html)]
2. <u>Stable Beluga 7B</u>: **"Stable Beluga 2"**. *Mahan, Dakota and Carlow, Ryan and Castricato, Louis and Cooper, Nathan and Laforte, Christian.*  2023. [[HuggingFace](https://huggingface.co/stabilityai/StableBeluga2)] 
3. <u>Fine-tuned BioGPT Guo et al.</u>: **"Improving Small Language Models on PubMedQA via Generative Data Augmentation"**. *Zhen Guo, Peiqi Wang, Yanwei Wang, Shangdi Yu.*  2023. [[Paper](https://arxiv.org/abs/2305.07804)]
4. <u>Financial SLMs</u>: **"Fine-tuning Smaller Language Models for Question Answering over Financial Documents"**. *Karmvir Singh Phogat Karmvir Singh Phogat, Sai Akhil Puranam, Sridhar Dasaratha, Chetan Harsha, Shashishekar Ramakrishna.*  2024. [[Paper](https://arxiv.org/abs/2408.12337)]
5. <u>ColBERT</u>: **"ColBERT Retrieval and Ensemble Response Scoring for Language Model Question Answering"**. *Alex Gichamba, Tewodros Kederalah Idris, Brian Ebiyau, Eric Nyberg, Teruko Mitamura.*  IEEE 2024. [[Paper](https://arxiv.org/abs/2408.10808)] 
6. <u>T-SAS</u>: **"Test-Time Self-Adaptive Small Language Models for Question Answering"**. *Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Hwang, Jong Park.* ACL 2023. [[Paper](https://aclanthology.org/2023.findings-emnlp.1033)] [[Github](https://github.com/starsuzi/T-SAS)]
7. <u>Rationale Ranking</u>: **"Answering Unseen Questions With Smaller Language Models Using Rationale Generation and Dense Retrieval"**. *Tim Hartill, Diana Benavides-Prado, Michael Witbrock, Patricia J. Riddle.*  2023. [[Paper](https://arxiv.org/abs/2308.04711)]

#### SLM in Coding

1. <u>Phi-3.5-mini</u>: **"Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone"**. *Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl, ..., Chunyu Wang, Guanhua Wang, Lijuan Wang et al.*  2024. [[Paper](https://arxiv.org/abs/2404.14219)] [[HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)] [[Website](https://azure.microsoft.com/en-us/products/phi)]
2. <u>TinyLlama</u>: **"TinyLlama: An Open-Source Small Language Model"**. *Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, Wei Lu.*  2024. [[Paper](https://arxiv.org/abs/2401.02385)] [[HuggingFace](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)] [[Chat Demo](https://huggingface.co/spaces/TinyLlama/tinyllama-chat)] [[Discord](https://discord.com/invite/74Wcx4j5Nb)]
3. <u>CodeLlama</u>: **"Code Llama: Open Foundation Models for Code"**. *Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Romain Sauvestre, Tal Remez, ..., Nicolas Usunier, Thomas Scialom, Gabriel Synnaeve.*  2024. [[Paper](https://arxiv.org/abs/2308.12950)] [[HuggingFace](https://huggingface.co/codellama)] 
4. <u>CodeGemma</u>: **"CodeGemma: Open Code Models Based on Gemma"**. *CodeGemma Team: Heri Zhao, Jeffrey Hui, Joshua Howland, Nam Nguyen, Siqi Zuo, Andrea Hu, Christopher A. Choquette-Choo, Jingyue Shen, Joe Kelley, Kshitij Bansal, ..., Kathy Korevec, Kelly Schaefer, Scott Huffman.*  2024. [[Paper](https://arxiv.org/abs/2406.11409)] [[HuggingFace](https://huggingface.co/google/codegemma-7b-it)] 

#### SLM in Recommendation

1. <u>PromptRec</u>: **"Could Small Language Models Serve as Recommenders? Towards Data-centric Cold-start Recommendations"**. *Xuansheng Wu, Huachi Zhou, Yucheng Shi, Wenlin Yao, Xiao Huang, Ninghao Liu.*  2024. [[Paper](https://arxiv.org/abs/2306.17256v4)] [[Github](https://github.com/JacksonWuxs/PromptRec)] 
2. <u>SLIM</u>: **"Can Small Language Models be Good Reasoners for Sequential Recommendation?"**. *Yuling Wang, Changxin Tian, Binbin Hu, Yanhua Yu, Ziqi Liu, Zhiqiang Zhang, Jun Zhou, Liang Pang, Xiao Wang.*  2024. [[Paper](https://arxiv.org/abs/2403.04260v1)]
3. <u>BiLLP</u>: **"Large Language Models are Learnable Planners for Long-Term Recommendation"**. *Wentao Shi, Xiangnan He, Yang Zhang, Chongming Gao, Xinyue Li, Jizhi Zhang, Qifan Wang, Fuli Feng.*  2024. [[Paper](https://arxiv.org/abs/2403.00843v2)] 
4. <u>ONCE</u>: **"ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models"**. *Qijiong Liu, Nuo Chen, Tetsuya Sakai, Xiao-Ming Wu.*  WSDM 2024. [[Paper](https://dl.acm.org/doi/10.1145/3616855.3635845)] [[Github](https://github.com/Jyonn/ONCE)]
5. <u>RecLoRA</u>: **"Lifelong Personalized Low-Rank Adaptation of Large Language Models for Recommendation"**. *Jiachen Zhu, Jianghao Lin, Xinyi Dai, Bo Chen, Rong Shan, Jieming Zhu, Ruiming Tang, Yong Yu, Weinan Zhang.*  2024. [[Paper](https://arxiv.org/abs/2408.03533v2)] 

#### SLM in Web Search

1. <u>Content encoder</u>: **"Pre-training Tasks for Embedding-based Large-scale Retrieval"**. *Wei-Cheng Chang, Felix X. Yu, Yin-Wen Chang, Yiming Yang, Sanjiv Kumar.*  ICLR 2020. [[Paper](https://arxiv.org/abs/2002.03932)]
2. <u>Poly-encoders</u>: **"Poly-encoders: Transformer Architectures and Pre-training Strategies for Fast and Accurate Multi-sentence Scoring"**. *Samuel Humeau, Kurt Shuster, Marie-Anne Lachaux, Jason Weston.*  ICLR 2020. [[Paper](https://arxiv.org/abs/1905.01969)]
3. <u>Twin-BERT</u>: **"TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval"**. *Wenhao Lu, Jian Jiao, Ruofei Zhang.*  2020. [[Paper](https://arxiv.org/abs/2002.06275)]
4. <u>H-ERNIE</u>: **"H-ERNIE: A Multi-Granularity Pre-Trained Language Model for Web Search"**. *Xiaokai Chu, Jiashu Zhao, Lixin Zou, Dawei Yin.*  SIGIR 2022. [[Paper](https://dl.acm.org/doi/10.1145/3477495.3531986)]
5. <u>Ranker</u>: **"Passage Re-ranking with BERT"**. *Rodrigo Nogueira, Kyunghyun Cho.*  2019. [[Paper](https://arxiv.org/abs/1901.04085)] [[Github](https://github.com/nyu-dl/dl4marco-bert)]
6. <u>Rewriter</u>: **"Query Rewriting for Retrieval-Augmented Large Language Models"**. *Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan.*  EMNLP2023. [[Paper](https://arxiv.org/abs/2305.14283)] [[Github](https://github.com/xbmxb/RAG-query-rewriting)]

#### SLM in Mobile-device

1. <u>Octopus</u>: **"Octopus: On-device language model for function calling of software APIs"**. *Wei Chen, Zhiyuan Li, Mingyuan Ma.*  2024. [[Paper](https://arxiv.org/abs/2404.01549)] [[HuggingFace](https://huggingface.co/NexaAIDev/Octopus-v1-gemma-7B)]
2. <u>MobileAgent</u>: **"Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration"**. *Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, Jitao Sang.*  2024. [[Paper](https://arxiv.org/abs/2406.01014)] [[Github](https://github.com/X-PLUG/MobileAgent)] [[HuggingFace](https://huggingface.co/spaces/junyangwang0410/Mobile-Agent)]
3. <u>Revolutionizing Mobile Interaction</u>: **"Revolutionizing Mobile Interaction: Enabling a 3 Billion Parameter GPT LLM on Mobile"**. *Samuel Carreira, Tomás Marques, José Ribeiro, Carlos Grilo.*  2023. [[Paper](https://arxiv.org/abs/2310.01434)] 
4. <u>AutoDroid</u>: **"AutoDroid: LLM-powered Task Automation in Android"**. *Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, Yunxin Liu.*  2023. [[Paper](https://arxiv.org/abs/2308.15272)]
5. <u>On-device Agent for Text Rewriting</u>: **"Towards an On-device Agent for Text Rewriting"**. *Yun Zhu, Yinxiao Liu, Felix Stahlberg, Shankar Kumar, Yu-hui Chen, Liangchen Luo, Lei Shu, Renjie Liu, Jindong Chen, Lei Meng.*  2023. [[Paper](https://arxiv.org/abs/2308.11807)]

### On-device Deployment Optimization Techniques

#### Memory Efficiency Optimization

1. <u>EDGE-LLM</u>: **"EDGE-LLM: Enabling Efficient Large Language Model Adaptation on Edge Devices via Layerwise Unified Compression and Adaptive Layer Tuning and Voting"**. *Zhongzhi Yu, Zheng Wang, Yuhan Li, Haoran You, Ruijie Gao, Xiaoya Zhou, Sreenidhi Reedy Bommu, Yang Katie Zhao, Yingyan Celine Lin.*  2024. [[Paper](https://arxiv.org/abs/2406.15758)] [[Github](https://github.com/GATECH-EIC/Edge-LLM)]
2. <u>LLM-PQ</u>: **"LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization"**. *Juntao Zhao, Borui Wan, Yanghua Peng, Haibin Lin, Chuan Wu.*  2024. [[Paper](https://arxiv.org/abs/2403.01136)] [[Github](https://github.com/tonyzhao-jt/LLM-PQ?tab=readme-ov-file)]
3. <u>AWQ</u>: **"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"**. *Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han.*  MLSys 2024. [[Paper](https://arxiv.org/abs/2306.00978)] [[Github](https://github.com/mit-han-lab/llm-awq)]
4. <u>MobileAIBench</u>: **"MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases"**. *Rithesh Murthy, Liangwei Yang, Juntao Tan, Tulika Manoj Awalgaonkar, Yilun Zhou, Shelby Heinecke, Sachin Desai, Jason Wu, Ran Xu, Sarah Tan, Jianguo Zhang, Zhiwei Liu, Shirley Kokane, Zuxin Liu, Ming Zhu, Huan Wang, Caiming Xiong, Silvio Savaresel.*  2024. [[Paper](https://arxiv.org/abs/2406.10290)] [[Github](https://github.com/XiaoMi/mobile-ai-bench)]
5. <u>MobileLLM</u>: **"MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases"**. *Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra.* ICML 2024. [[Paper](https://arxiv.org/abs/2402.14905)] [[Github](https://github.com/facebookresearch/MobileLLM)] [[HuggingFace](https://huggingface.co/papers/2402.14905)]
6. <u>EdgeMoE</u>: **"EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models"**. *Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, Mengwei Xu.*  2023. [[Paper](https://arxiv.org/abs/2308.14352)] [[Github](https://github.com/sharc-lab/Edge-MoE)] 
7. <u>GEAR</u>: **"GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM"**. *Hao Kang, Qingru Zhang, Souvik Kundu, Geonhwa Jeong, Zaoxing Liu, Tushar Krishna, Tuo Zhao.*  2024. [[Paper](https://arxiv.org/abs/2403.05527)] [[Github](https://github.com/opengear-project/GEAR)]
8. <u>DMC</u>: **"Dynamic Memory Compression: Retrofitting LLMs for Accelerated Inference"**. *Piotr Nawrot, Adrian Łańcucki, Marcin Chochowski, David Tarjan, Edoardo M. Ponti.*  2024. [[Paper](https://arxiv.org/abs/2403.09636)]
9. <u>Transformer-Lite</u>: **"Transformer-Lite: High-efficiency Deployment of Large Language Models on Mobile Phone GPUs"**. *Luchang Li, Sheng Qian, Jie Lu, Lunxi Yuan, Rui Wang, Qin Xie.*  2024. [[Paper](https://arxiv.org/abs/2403.20041)]
10. <u>LLMaaS</u>: **"LLM as a System Service on Mobile Devices"**. *Wangsong Yin, Mengwei Xu, Yuanchun Li, Xuanzhe Liu.*  2024. [[Paper](https://arxiv.org/abs/2403.11805)]

#### Runtime Efficiency Optimization

1. <u>EdgeMoE</u>: **"EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models"**. *Rongjie Yi, Liwei Guo, Shiyun Wei, Ao Zhou, Shangguang Wang, Mengwei Xu.*  2023. [[Paper](https://arxiv.org/abs/2308.14352)] [[Github](https://github.com/sharc-lab/Edge-MoE)] 
2. <u>LLMCad</u>: **"LLMCad: Fast and Scalable On-device Large Language Model Inference"**. *Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, Xuanzhe Liu.*  2023. [[Paper](https://arxiv.org/abs/2309.04255)]
3. <u>LinguaLinked</u>: **"LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices"**. *Junchen Zhao, Yurun Song, Simeng Liu, Ian G. Harris, Sangeetha Abdu Jyothi.*  2023 [[Paper](https://arxiv.org/abs/2312.00388)]

### SLMs enhance LLMs

#### SLMs for LLM Calibration and Hallucination Detection

1. **Calibrating Large Language Models Using Their Generations Only.** *Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, Seong Joon Oh*. ACL 2024 Long, [[pdf]](https://aclanthology.org/2024.acl-long.824/) [[code]](https://github.com/parameterlab/apricot)
2. **Pareto Optimal Learning for Estimating Large Language Model Errors.** *Theodore Zhao, Mu Wei, J. Samuel Preston, Hoifung Poon*. ACL 2024 Long, [[pdf]](https://aclanthology.org/2024.acl-long.566/)
3. **The Internal State of an LLM Knows When It’s Lying.** *Amos Azaria, Tom Mitchell*. EMNLP 2023 Findings. [[pdf]](https://aclanthology.org/2023.findings-emnlp.68/)
4. **Small agent can also rock! empowering small language models as hallucination detector.** _Xiaoxue Cheng, Junyi Li, Wayne Xin Zhao, Hongzhi Zhang, Fuzheng Zhang, Di Zhang, Kun Gai, Ji-Rong Wen._ EMNLP 2024 Long. [[pdf]](https://aclanthology.org/2024.emnlp-main.809/)
5. **Reconfidencing llms from the grouping loss perspective.** _Lihu Chen, Alexandre Perez-Lebel, Fabian M. Suchanek, Gaël Varoquaux._ EMNLP 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-emnlp.85/)

#### SLMs for LLM RAG

1. **Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs.** *Jiejun Tan, Zhicheng Dou, Yutao Zhu, Peidong Guo, Kun Fang, Ji-Rong Wen.* ACL 2024 Long.  [[pdf]](https://aclanthology.org/2024.acl-long.242/) [[code]](https://github.com/plageon/SlimPlm) [[huggingface]](https://huggingface.co/zstanjj/SlimPLM-Query-Rewriting)
2. **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.** *Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi.* ICLR 2024 Oral. [[pdf]](https://openreview.net/forum?id=hSyW5go0v8) [[huggingface]](https://huggingface.co/papers/2310.11511) [[code]](https://github.com/AkariAsai/self-rag) [[website]](https://selfrag.github.io/) [[model]](https://huggingface.co/selfrag/selfrag_llama2_7b) [[data]](https://huggingface.co/datasets/selfrag/selfrag_train_data) 
3. **LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression.** *Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, Lili Qiu.* ICLR 2024 Workshop ME-FoMo Poster. [[pdf]](https://openreview.net/forum?id=9YvfRrpmyw) 
4. **Corrective Retrieval Augmented Generation.** *Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling.* arXiv 2024.1. [[pdf]](https://arxiv.org/abs/2401.15884) [[code]](https://github.com/HuskyInSalt/CRAG)
5. **Self-Knowledge Guided Retrieval Augmentation for Large Language Models.** *Yile Wang, Peng Li, Maosong Sun, Yang Liu.* EMNLP 2023 Findings. [[pdf]](https://aclanthology.org/2023.findings-emnlp.691/) [[code]](https://github.com/THUNLP-MT/SKR)
6. **In-Context Retrieval-Augmented Language Models.** *Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, Yoav Shoham.* TACL 2023. [[pdf]](https://aclanthology.org/2023.tacl-1.75/) [[code]](https://github.com/AI21Labs/in-context-ralm)
7. **RA-ISF: Learning to Answer and Understand from Retrieval Augmentation via Iterative Self-Feedback.** _Liu, Yanming and Peng, Xinyue and Zhang, Xuhong and Liu, Weihao and Yin, Jianwei and Cao, Jiannan and Du, Tianyu._ ACL 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-acl.281/)
8. **Less is More: Making Smaller Language Models Competent Subgraph Retrievers for Multi-hop {KGQA}.** _Wenyu Huang, Guancheng Zhou, Hongru Wang, Pavlos Vougiouklis, Mirella Lapata, Jeff Z. Pan._ EMNLP 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-emnlp.927/)

#### SLMs for LLM Reasoning
1. _Canwen Xu, Yichong Xu, Shuohang Wang, Yang Liu, Chenguang Zhu, and Julian McAuley._ **Small models are valuable plug-ins for large language models.** ACL 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-acl.18/)
2. _Linyi Yang, Shuibai Zhang, Zhuohao Yu, Guangsheng Bao, Yidong Wang, Jindong Wang, Ruochen Xu, Wei Ye, Xing Xie, Weizhu Chen, and Yue Zhang._ **Supervised Knowledge Makes Large Language Models Better In-context Learners.** ICLR 2024 Poster. [[pdf]](https://openreview.net/forum?id=bAMPOUF227)
3. _Zhuofeng Wu, He Bai, Aonan Zhang, Jiatao Gu, VG Vydiswaran, Navdeep Jaitly, and Yizhe Zhang._  **Divide-or-Conquer? Which Part Should You Distill Your LLM?** EMNLP 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-emnlp.145/)

#### SLMs for alleviating Copyright and Privacy of LLMs
1. _Tianlin Li, Qian Liu, Tianyu Pang, Chao Du, Qing Guo, Yang Liu, and Min Lin._ **Purifying large language models by ensembling a small language model.** arXiv 2024. [[pdf]](https://arxiv.org/abs/2402.14845)

#### SLMs for extracting LLM prompts
1. _Yiming Zhang, Nicholas Carlini, and Daphne Ippolito._ **Effective Prompt Extraction from Language Models.** COLM 2024 [[pdf]](https://openreview.net/forum?id=0o95CVdNuz#discussion)
2. _Zeyang Sha and Yang Zhang._ Prompt stealing attacks against large language models. arXiv (2024). [[pdf]](https://arxiv.org/abs/2402.12959)
3. _Collin Zhang, John X Morris, and Vitaly Shmatikov._ **Extracting Prompts by Inverting LLM Outputs.** [[pdf]](https://aclanthology.org/2024.emnlp-main.819/)


#### SLMs for Fine-tuning LLMs
1. _Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, and Christopher D Manning._ 2024. **An Emulator for Fine-tuning Large Language Models using Small Language Models.** ICLR 2024. [[pdf]](https://openreview.net/forum?id=Eo7kv0sllr)
2. _Alisa Liu, Xiaochuang Han, Yizhong Wang, Yulia Tsvetkov, Yejin Choi, and Noah A Smith._ 2024. **Tuning language models by proxy.** COLM 2024. [[pdf]](https://openreview.net/forum?id=dribhnhm1i)
3. _Dheeraj Mekala, Alex Nguyen, and Jingbo Shang._ 2024. **Smaller language models are capable of selecting instruction-tuning training data for larger language models.** ACL 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-acl.623/)
4. _Yongheng Deng, Ziqing Qiao, Ju Ren, Yang Liu, and Yaoxue Zhang._ 2023. **Mutual enhancement of large and small language models with cross-silo knowledge transfer.** arXiv 2023. [[pdf]](https://arxiv.org/abs/2312.05842)
5. **SmallToLarge (S2L): Scalable Data Selection for Fine-tuning Large Language Models by Summarizing Training Trajectories of Small Models.** _Yu Yang · Siddhartha Mishra · Jeffrey Chiang · Baharan Mirzasoleiman._ NIPS 2024 Poster. [[pdf]](https://neurips.cc/virtual/2024/poster/95679)
6. **Weak-to-Strong Search: Align Large Language Models via Searching over Small Language Models.** _Zhanhui Zhou · Zhixuan Liu · Jie Liu · Zhichen Dong · Chao Yang · Yu Qiao._ NIPS 2024 Poster. [[pdf]](https://neurips.cc/virtual/2024/poster/94341)

#### SLMs for LLM safety
1. **Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations.** _Meta_ arXiv 2024 [[pdf]](https://arxiv.org/abs/2312.06674)
2. **SLM as Guardian: Pioneering AI Safety with Small Language Model.** _Ohjoon Kwon, Donghyeon Jeon, Nayoung Choi, Gyu-Hwung Cho, Hwiyeol Jo, Changbong Kim, Hyunwoo Lee, Inho Kang, Sun Kim, Taiwoo Park._ EMNLP 2024. [[pdf]](https://aclanthology.org/2024.emnlp-industry.99/)

#### SLM for LLM Evaluation
1. _Kun Zhao, Bohao Yang, Chen Tang, Chenghua Lin, and Liang Zhan_. 2024. **SLIDE: A Framework Integrating Small and Large Language Models for Open-Domain Dialogues Evaluation**. ACL 2024 Findings. [[pdf]](https://aclanthology.org/2024.findings-acl.911/)
2. **Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation.** _Lorenz Kuhn, Yarin Gal, Sebastian Farquhar._ ICLR 2023. [[pdf]](https://openreview.net/forum?id=VD-AYtP0dve)
3. **Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models.** _Potsawee Manakul, Adian Liusie, Mark Gales._ EMNLP 2023 Main. [[pdf]](https://aclanthology.org/2023.emnlp-main.557/)
4. **Proxylm: Predicting language model performance on multilingual tasks via proxy models.** _David Anugraha, Genta Indra Winata, Chenyue Li, Patrick Amadeus Irawan, En-Shiun Annie Lee._ arXiv 2024. [[pdf]](https://arxiv.org/abs/2406.09334)
5. **Factscore: Fine-grained atomic evaluation of factual precision in long-form text generation.** _Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi._ EMNLP 2023 Main. [[pdf]](https://aclanthology.org/2023.emnlp-main.741/)
6. **Look before you leap: An exploratory study of uncertainty measurement for large language models.** _Yuheng Huang, Jiayang Song, Zhijie Wang, Shengming Zhao, Huaming Chen, Felix Juefei-Xu, Lei Ma_ arXiv 2023. [[pdf]](https://arxiv.org/abs/2307.10236)



### Synergy between SLMs and LLMs
1. **CoGenesis: A Framework Collaborating Large and Small Language Models for Secure Context-Aware Instruction Following.** *Kaiyan Zhang, Jianyu Wang, Ermo Hua, Biqing Qi, Ning Ding, Bowen Zhou.* arXiv 2024.6..  [[pdf]](https://arxiv.org/abs/2403.03129)
2. **When Large Language Model Agents Meet 6G Networks: Perception, Grounding, and Alignment.** *Minrui Xu; Dusit Niyato; Jiawen Kang; Zehui Xiong; Shiwen Mao; Zhu Han.* IEEE Wireless Communications, 2024.  [[pdf]](https://ieeexplore.ieee.org/abstract/document/10648594)
3. **Think Big, Generate Quick: LLM-to-SLM for Fast Autoregressive Decoding.** *Benjamin Bergner, Andrii Skliar, Amelie Royer, Tijmen Blankevoort, Yuki Asano, Babak Ehteshami Bejnordi.* arXiv, 2024.7.  [[pdf]](https://arxiv.org/abs/2402.16844)
4. **Synergy-of-Thoughts: Eliciting Efficient Reasoning in Hybrid Language Models.** *Yu Shang, Yu Li, Fengli Xu, Yong Li.* arXiv, 2024.8.  [[pdf]](https://arxiv.org/abs/2402.02563)
5. **Hybrid SLM and LLM for Edge-Cloud Collaborative Inference.** *Zixu Hao, Huiqiang Jiang, Shiqi Jiang, Ju Ren, Ting Cao.* EdgeFM 2024.  [[pdf]](https://dl.acm.org/doi/abs/10.1145/3662006.3662067)
6. **LLMCad: Fast and Scalable On-device Large Language Model Inference.** *Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, Xuanzhe Liu.* arXiv 2023.9.  [[pdf]](https://arxiv.org/abs/2309.04255)
7. **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.** *Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, Christopher Potts.* arXiv 2023.10.  [[pdf]](https://arxiv.org/abs/2310.03714)
8. **Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!.** *Yubo Ma, Yixin Cao, YongChing Hong, Aixin Sun.* arXiv 2023.10.  [[pdf]](https://arxiv.org/abs/2303.08559)
9. **Mutual Enhancement of Large and Small Language Models with Cross-Silo Knowledge Transfer.** *Yongheng Deng, Ziqing Qiao, Ju Ren, Yang Liu, Yaoxue Zhang.* arXiv 2023.12.  [[pdf]](https://arxiv.org/abs/2312.05842) 
10. **Small LLMs Are Weak Tool Learners: A Multi-LLM Agent.** *Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, Fei Huang.* EMNLP 2024 Main.  [[pdf]](https://aclanthology.org/2024.emnlp-main.929/) 
11. **Synergizing Large Language Models and Pre-Trained Smaller Models for Conversational Intent Discovery.** *Jinggui Liang, Lizi Liao, Hao Fei, Jing Jiang.* ACL 2024 Findings.  [[pdf]](https://aclanthology.org/2024.findings-acl.840/) 
12. **Improving Large Models with Small Models: Lower Costs and Better Performance.** *Dong Chen, Shuo Zhang, Yueting Zhuang, Siliang Tang, Qidong Liu, Hua Wang, Mingliang Xu.* arXiv 2024.6.  [[pdf]](https://arxiv.org/abs/2406.15471)

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=FairyFali/SLMs-Survey&type=Date)

<!-- Insertion Template: 0. <u>Model</u>: **"Title"**. *Name et al.*  202X. [[Paper]()] [[Github]()] [[HuggingFace]()] -->

