for month in $(seq -f '%02g' 12); do
    for day in $(seq -f '%02g' 31); do
        wget https://docs.misoenergy.org/marketreports/2019${month}${day}_rt_lmp_final.csv
    done
done
