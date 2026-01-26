python aggregate_results.py --dataset PACS --backbone resnet18 --seeds 0 1 2 3 4 --main_exp_name imgaug_and_canny_training_all --cv_exp_names imgaug_and_canny_training_first imgaug_and_canny_training_second
python aggregate_results.py --dataset PACS --backbone resnet18 --seeds 0 1 2 3 4 --main_exp_name original_and_canny_training   
python aggregate_results.py --dataset PACS --backbone resnet18 --seeds 0 1 2 3 4 --main_exp_name original-only_training   
python aggregate_results.py --dataset PACS --backbone caffenet --seeds 0 1 2 3 4 --main_exp_name imgaug_and_canny_training_all --cv_exp_names imgaug_and_canny_training_first imgaug_and_canny_training_second
python aggregate_results.py --dataset PACS --backbone caffenet --seeds 0 1 2 3 4 --main_exp_name original_and_canny_training   
python aggregate_results.py --dataset PACS --backbone caffenet --seeds 0 1 2 3 4 --main_exp_name original-only_training      
python aggregate_results.py --dataset PACS --backbone vit_small --seeds 0 1 2 3 4 --main_exp_name imgaug_and_canny_training_all --cv_exp_names imgaug_and_canny_training_first imgaug_and_canny_training_second
python aggregate_results.py --dataset PACS --backbone vit_small --seeds 0 1 2 3 4 --main_exp_name original_and_canny_training   
python aggregate_results.py --dataset PACS --backbone vit_small --seeds 0 1 2 3 4 --main_exp_name original-only_training     
   
python make_scatter_plots.py --dataset PACS
