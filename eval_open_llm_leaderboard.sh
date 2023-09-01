MODEL = $1
mkdir -p results/$MODEL
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,dtype=float16 --tasks arc_challenge --batch_size 1 --no_cache --write_out --output_path results/$MODEL/arc_challenge_25shot.json --device cuda --num_fewshot 25
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,dtype=float16 --tasks hellaswag --batch_size 1 --no_cache --write_out --output_path results/$MODEL/hellaswag_10shot.json --device cuda --num_fewshot 10
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,dtype=float16 --tasks hendrycksTest-* --batch_size 1 --no_cache --write_out --output_path results/$MODEL/mmlu_5shot.json --device cuda --num_fewshot 5
python main.py --model hf-causal-experimental --model_args pretrained=$MODEL,dtype=float16 --tasks truthfulqa_mc --batch_size 1 --no_cache --write_out --output_path results/$MODEL/truthfulqa_0shot.json --device cuda

