# Executing train.py
CUDA_VISIBLE_DEVICES=6,7,4,5,2,3 python -m torch.distributed.launch --nproc_per_node=6 wcl.py
train_exit_code=$?
if [ $train_exit_code -ne 0 ]; then
    echo "Error, Exit Code: $train_exit_code"
    exit $train_exit_code
fi

# Executing test.py
python ddt_dino_sig.py
test_exit_code=$?
if [ $test_exit_code -ne 0 ]; then
    echo "Error, Exit Code: $test_exit_code"
    exit $test_exit_code
fi

# Executing measure.py
python measure.py
measure_exit_code=$?
if [ $measure_exit_code -ne 0 ]; then
    echo "Error, Exit Code: $measure_exit_code"
    exit $measure_exit_code
fi

echo "The End!"