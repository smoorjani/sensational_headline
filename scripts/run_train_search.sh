declare -a mode_arr=("tag" "emb")
# declare -a mode_arr=("tag")
declare -a discrim_arr=("bart" "roberta")
# declare -a discrim_arr=("bart")

gamma=(0.05 0.1 0.2)

# for i in {0..1}
for g in "${gamma[@]}"
do 
    for i in {1..1}
    do 
        for j in "${mode_arr[@]}"
        do

            export USE_DISCRIMINATOR=$i
            export MODE=$j
            export GAMMA=$g

            if [ $i == 1 ]; then
                for k in "${discrim_arr[@]}"
                do
                    
                    if [ $k == "bart" ]; then
                        export DISCRIM="/projects/bblr/smoorjani/control_tuning/speed_control/bart/checkpoint_6_0_False_5e-05_2_0.8_3_0.001"
                    else
                        export DISCRIM="/projects/bblr/smoorjani/control_tuning/speed_control/roberta/checkpoint_4_0_False_5e-05_2_0.8_3_0.001"
                    fi

                    echo "Running with settings USE_DISCRIMINATOR=$i and MODE=$j and DISCRIMINATOR=$k"
                    sbatch --export=ALL scripts/train.slurm
                done
            else
                echo "Running with settings USE_DISCRIMINATOR=$i and MODE=$j"
                sbatch --export=ALL scripts/train.slurm
            fi

        done
    done
done    
