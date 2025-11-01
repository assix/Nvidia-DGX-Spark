🚀 Small Language Model (SLM) Benchmark

This script benchmarks the performance of various small language models (under 4 billion parameters) using the Hugging Face transformers library on a CUDA-enabled GPU. It's designed to provide insights into load times, VRAM consumption, and inference speed on specific hardware.

✨ Features

This benchmark measures the following key metrics for each model:

    Model Loading Time: Time taken to load the model and tokenizer into memory.

    Disk Size: Estimated disk space occupied by the model/tokenizer files (calculated by saving the loaded model).

    Peak VRAM (Load): Maximum GPU memory allocated after loading the model.

    Peak VRAM (Generation): Maximum GPU memory allocated during the text generation phase.

    Generation Speed: Inference speed measured in tokens per second.

    Parameter Count: The approximate number of parameters (in billions).

    Knowledge Cutoff: The approximate date of the model's training data cutoff.

🛠️ Requirements

    Python 3.x

    PyTorch (with CUDA support)

    Hugging Face transformers

    Hugging Face accelerate (for optimized loading)

    Hugging Face huggingface_hub (for the CLI)

    A CUDA-enabled GPU is required as the script uses torch.cuda for VRAM measurements.

⚡ Quick Start

1. Clone the Repository

Bash

git clone <your-repo-url>
cd <your-repo-directory>

2. Install Requirements

Install the necessary Python libraries.
Bash

pip install torch transformers accelerate huggingface_hub

    Important: Ensure your PyTorch installation matches your system's CUDA version. See the PyTorch Get Started page for the correct command for your environment.

3. Log in to Hugging Face (Optional)

This is required if you are benchmarking gated models like Llama or Gemma. You must first accept their license terms on the respective Hugging Face model pages.
Bash

huggingface-cli login

4. Run the Benchmark

Bash

python BenchmarkSmallLanguageModels.py

📊 Models Benchmarked

The following models are included by default:

    TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.10B)

    meta-llama/Llama-3.2-1B-Instruct (1.23B)

    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.50B)

    Qwen/Qwen1.5-1.8B-Chat (1.80B)

    google/gemma-2b-it (2.50B)

    microsoft/Phi-3-mini-4k-instruct (3.80B)

    Want to test other models? You can easily add, remove, or change models by editing the MODELS_TO_BENCHMARK dictionary within the BenchmarkSmallLanguageModels.py script.

📄 Output

The script will:

    Print benchmark progress and individual model summaries to the console.

    Print a final comparison table to the console, sorted by performance (Tokens/sec, highest first).

    Save the same final comparison table to a file named benchmark_results.txt.

Example Results (DGX Spark - Blackwell GPU, bfloat16)

Model	Owner	Params (B)	Size (GB)	Tokens/sec	Peak VRAM (GB)	Load Time(s)	Knowledge Cutoff
TinyLlama-1.1B	TinyLlama	1.10B	2.05	59.68	2.06	14.01	N/A
Llama-3.2-1B	Meta	1.23B	2.32	57.49	2.32	16.98	Dec 2023
DeepSeek-1.5B	DeepSeek AI	1.50B	3.32	46.58	3.33	22.49	July 2024
Qwen1.5-1.8B	Alibaba	1.80B	3.44	42.98	3.51	27.69	Jan 2024
Gemma-2B-IT	Google	2.50B	4.70	32.82	4.68	25.60	June 2024
Phi-3-mini-4k	Microsoft	3.80B	7.12	17.34	7.18	50.38	Oct 2023

🗒️ Notes

    Results are highly dependent on the specific hardware (GPU, CPU, RAM), software versions (CUDA, PyTorch), and model precision (this script uses bfloat16).

    VRAM measurements rely on torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated().

    Disk size is an estimate calculated by temporarily saving the loaded model. The actual download size may differ.

    The Phi-3 model required specific workarounds (attn_implementation='eager', use_cache=False) due to potential incompatibilities. Ensure your transformers library is up-to-date.

📜 License

Apache 2.0
