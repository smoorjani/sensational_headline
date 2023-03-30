echo "Starting speed computations"

export PYTHONPATH="${PYTHONPATH}:/u/smoorjani/sensational_headline"

rm control_scores.csv

FILES="/projects/bblr/smoorjani/control_tuning/generations/speeds/*.txt"
GENERATIONS_DIR="/projects/bblr/smoorjani/control_tuning/generations"
for f in $FILES
do
    echo "Processing $f"
    python scripts/run_control_evaluation.py --pred_file ${f//"$GENERATIONS_DIR"} 
done