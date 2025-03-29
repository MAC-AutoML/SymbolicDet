# From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning

[![Arxiv](https://img.shields.io/badge/Arxiv-2502.05843-red)](https://arxiv.org/abs/2502.05843)
![License](https://img.shields.io/badge/license-MIT-blue)

<font size=3><div align='center' > [[🍎 Project Page](https://ved-sr.github.io/)] [[📖 arXiv 
Paper](https://arxiv.org/abs/2502.05843)] </div></font>

## 🚀 Usage

### Running the SymbolicDet Framework

You can run the framework either using the Python script directly or using the provided bash script:

#### Using Bash Script (Recommended)

```bash
# Give execution permission
chmod +x run_sr.sh

# Set API key (required for LLM functionality)
# Option 1: Using environment variable
export SR_API_KEY="your_api_key"

# Option 2: Using command line argument
./run_sr.sh -k "your_api_key"

# Run with default settings (LLM enabled)
./run_sr.sh

# Run without LLM assistance
./run_sr.sh --no-llm

# Run with custom config file
./run_sr.sh -c path/to/config.yaml

# Run with multiple options
./run_sr.sh --no-llm -c path/to/config.yaml
./run_sr.sh -k "your_api_key" -c path/to/config.yaml
```

#### Script Options:
- `-h, --help`: Show help information
- `--no-llm`: Disable LLM functionality
- `-c, --config <path>`: Specify configuration file path
- `-k, --api-key <key>`: Set API key (can also be set via SR_API_KEY environment variable)

#### Requirements:

- Python 3.7+
- Required packages: see `requirements.txt`
- API key for LLM service (when using LLM functionality)

## ✏️ Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and citation 📝:

```
@misc{zeng2025objectseventsunlockingcomplex,
      title={From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning}, 
      author={Yuhui Zeng and Haoxiang Wu and Wenjie Nie and Xiawu Zheng and Guangyao Chen and Yunhang Shen and Jun Peng and Yonghong Tian and Rongrong Ji},
      year={2025},
      eprint={2502.05843},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.05843}, 
}
```
