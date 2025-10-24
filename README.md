Small Language Model (SLM) Benchmark

This script benchmarks the performance of various small language models (under 4 billion parameters) using the Hugging Face transformers library on a CUDA-enabled GPU. It's designed to provide insights into load times, VRAM consumption, and inference speed on specific hardware.

Features

Measures the following metrics for each model:

Model Loading Time: Time taken to load the model and tokenizer into memory.

Disk Size: Estimated disk space occupied by the model and tokenizer files (calculated by temporarily saving the loaded model).

Peak VRAM (Load): Maximum GPU memory allocated after loading the model.

Peak VRAM (Generation): Maximum GPU memory allocated during the text generation phase.

Generation Speed: Inference speed measured in tokens per second.

Parameter Count: The approximate number of parameters in billions (B).

Knowledge Cutoff: The approximate date of the model's training data cutoff.

Requirements

Python 3.x

PyTorch (torch): Ensure a version compatible with your CUDA toolkit is installed. (See PyTorch Get Started)

Hugging Face transformers library (pip install transformers)

accelerate library (pip install accelerate) for optimized loading.

A CUDA-enabled GPU. This script specifically uses torch.cuda for VRAM measurements.

(Optional but Recommended) huggingface-cli for logging into Hugging Face Hub (pip install huggingface_hub[cli]).

Models Benchmarked

The following models are included in the benchmark by default:

TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.10B)

meta-llama/Llama-3.2-1B-Instruct (1.23B)

deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (1.50B)

Qwen/Qwen1.5-1.8B-Chat (1.80B)

google/gemma-2b-it (2.50B)

microsoft/Phi-3-mini-4k-instruct (3.80B)

You can easily modify the MODELS_TO_BENCHMARK dictionary within the BenchmarkSmallLanguageModels.py script to add, remove, or change models.

Usage

Clone the repository:

git clone <your-repo-url>
cd <your-repo-directory>


Install requirements:

pip install torch torchvision torchaudio transformers accelerate huggingface_hub
# Ensure torch install matches your CUDA version


Log in to Hugging Face (Required for Gated Models):
If benchmarking gated models like Llama or Gemma, you need to:

Accept their license terms on the respective Hugging Face model pages.

Log in via the terminal:

huggingface-cli login


Run the script:

python BenchmarkSmallLanguageModels.py


Output

The script will:

Print the benchmark progress and individual model summaries to the console.

Print a final comparison table to the console, sorted by performance (Tokens/sec, highest first).

Save the same final comparison table to a file named benchmark_results.txt in the script's directory.

Example Results (DGX Spark - Blackwell GPU, bfloat16)

--- Final Benchmark Comparison (Sorted by Performance) ---
Model              | Owner        | Params (B)  | Size (GB)  | Tokens/sec   | Peak VRAM(GB)   | Load Time(s)   | Knowledge Cutoff
-------------------------------------------------------------------------------------------------------------------------------------
TinyLlama-1.1B     | TinyLlama    | 1.10B       | 2.05       | 59.68        | 2.06            | 14.01          | N/A
Llama-3.2-1B       | Meta         | 1.23B       | 2.32       | 57.49        | 2.32            | 16.98          | Dec 2023
DeepSeek-1.5B      | DeepSeek AI  | 1.50B       | 3.32       | 46.58        | 3.33            | 22.49          | July 2024
Qwen1.5-1.8B       | Alibaba      | 1.80B       | 3.44       | 42.98        | 3.51            | 27.69          | Jan 2024
Gemma-2B-IT        | Google       | 2.50B       | 4.70       | 32.82        | 4.68            | 25.60          | June 2024
Phi-3-mini-4k      | Microsoft    | 3.80B       | 7.12       | 17.34        | 7.18            | 50.38          | Oct 2023
-------------------------------------------------------------------------------------------------------------------------------------


Notes

Benchmark results are highly dependent on the specific hardware (GPU, CPU, RAM), software versions (CUDA, PyTorch, transformers), and model quantization (this script uses bfloat16).

VRAM measurements rely on torch.cuda.memory_allocated() and torch.cuda.max_memory_allocated().

Disk size is an estimate calculated by temporarily saving the loaded model and tokenizer; the actual download size might differ due to caching and file formats.

The Phi-3 model required specific workarounds (attn_implementation='eager', use_cache=False) due to potential incompatibilities; ensure your transformers library is up-to-date.

License Apache 2.0
