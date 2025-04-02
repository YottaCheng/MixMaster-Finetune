conda activate mixing_env  
export QWEN_API_KEY="sk-3b986ed51abb4ed18aadde5d41e11397"
llamafactory-cli webui --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
运行prelabeled.py时 
conda activate snorkel_env
export DEEPSEEK_API_KEY ="sk-3511f72cb3324a36b42ac8dc91568769"
tensorboard --logdir=
conda activate audio_analyze