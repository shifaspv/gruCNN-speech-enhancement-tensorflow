if [ $# -ne 1 ]; then
    echo 'Error in parameters' 
    echo Usage: $0' model_id'
    #exit
fi

filelist=$2/*.wav

for file in $filelist
do
    filename=$(basename $file .wav)
    echo $filename
    python generate.py --config=config_params.json --model_id=$1 --noisy_speech_filename=$filename
done


