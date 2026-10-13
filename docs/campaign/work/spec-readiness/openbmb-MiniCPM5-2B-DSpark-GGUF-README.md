---
license: apache-2.0
language:
  - en
  - zh
library_name: transformers
pipeline_tag: text-generation
tags:
  - minicpm
  - minicpm5
  - llama
  - text-generation
  - long-context
  - tool-calling
  - on-device
  - edge-ai
datasets:
  - openbmb/Ultra-FineWeb
  - openbmb/UltraX-Preview
  - openbmb/Ultra-FineWeb-L3
  - openbmb/UltraData-Math
  - openbmb/UltraData-Code
  - openbmb/UltraData-SFT-2605
  - openbmb/UltraData-SFT-Agent-2609
  - openbmb/UltraData-RL-2609
---
<div align="center">
<img src="https://raw.githubusercontent.com/OpenBMB/MiniCPM/main/assets/minicpm_logo.png" width="500em" />
</div>

<p align="center">
<a href="https://arxiv.org/pdf/2506.07900" target="_blank">MiniCPM Tech Report</a> |
<a href="https://modelbest.feishu.cn/wiki/UtWxwcERfiRIpIkBOjuc3h9tn1D" target="_blank">MiniCPM Wiki(Chinese)</a> |
<a href="https://github.com/OpenBMB/MiniCPM" target="_blank">GitHub Repo</a> |
<a href="https://ultradata.openbmb.cn/" target="_blank">UltraData</a> |
<a href="https://huggingface.co/spaces/openbmb/MiniCPM5-2B-Demo" target="_blank">Online Demo</a>
</p>

# MiniCPM5-2B-DSpark-GGUF

GGUF version of [MiniCPM5-2B-DSpark](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark) for speculative decoding with [MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) in llama.cpp.

| File | Precision | Size |
| --- | --- | --- |
| `MiniCPM5-2.6B-DSpark.gguf` | BF16 | 653 MB |

## Usage with llama.cpp

```bash
llama-server \
  -m MiniCPM5-2B-Q4_K_M.gguf \
  -md MiniCPM5-2.6B-DSpark.gguf \
  --spec-type draft-dspark --spec-draft-n-max 7 \
  -ngl 99 -ngld 99 -fa on \
  -c 8192 --jinja --port 8080
```


## Limitations and Disclaimer

This model has no autonomous intent or legal personhood; its outputs are text generated from statistical patterns and may be inaccurate, biased, or offensive, and may be manipulated by carefully crafted prompts ("jailbreaks") into producing unintended content. Its responses on sensitive topics such as politics, health, finance, and law are not reviewed by experts and should not be treated as professional advice.

This model is provided "**AS IS**", without warranty of any kind, express or implied, and the developers are not liable for any damages arising from its use. Users must employ the model only for lawful, compliant, and ethical purposes, configure their own safeguards, and label AI-generated content where required; deliberate jailbreaking, injection attacks, or inducing harmful output is prohibited, and any such testing is at the user's own risk.

## License

This repository and MiniCPM model weights are released under the [Apache-2.0](https://github.com/OpenBMB/MiniCPM/blob/main/LICENSE) License.

## Citation

Please cite our paper if you find our work valuable:

```bibtex
@article{minicpm4,
  title={Minicpm4: Ultra-efficient llms on end devices},
  author={MiniCPM, Team},
  journal={arXiv preprint arXiv:2506.07900},
  year={2025}
}
```

