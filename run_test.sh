bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 1 --end_sprites 2  --do_l2
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 2 --end_sprites 3  --do_l2
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 3 --end_sprites 4  --do_l2
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 4 --end_sprites 5  --do_l2
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 5 --end_sprites 6  --do_l2

bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 1 --end_sprites 2  --do_contact
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 2 --end_sprites 3  --do_contact
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 3 --end_sprites 4  --do_contact
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 4 --end_sprites 5  --do_contact
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 5 --end_sprites 6  --do_contact

bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 1 --end_sprites 2  --do_sparse
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 2 --end_sprites 3  --do_sparse
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 3 --end_sprites 4  --do_sparse
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 4 --end_sprites 5  --do_sparse
bsub -n 5 -W 24:00  -R "rusage[mem=5000]" python3 algo_test.py --start_sprites 5 --end_sprites 6  --do_sparse

