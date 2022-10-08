echo "================================================================================================"
echo "Please run the script as: "
echo "bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]"
echo "For example: bash train_standalone.sh /home/x3d /home/publicfile/kinetics-400"
echo "================================================================================================"
set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash train_standalone.sh [PROJECT_PATH] [DATA_PATH]"
exit 1
fi

PYTHON_PATH=$1
DATA_PATH=$2

export PYTHONPATH=$PYTHON_PATH
python $PYTHON_PATH/src/example/x3d_kinetics400_train.py --data_url $DATA_PATH \
    >  train_standalone.log 2>&1


