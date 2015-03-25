echo "== TEST SCRIPT FOR PARALLEL SOM =="
echo -e "\t== Base test =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 32 x 32 map \t5120 x 3 vectors"
#./parallel_SOM datasets/input_5120_3_gauss.dat 32 1 > datasets/test_base_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\n\t== Test 1 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 32 x 32 map \t8192 x 3 vectors"
#./parallel_SOM datasets/input_8192_3_gauss.dat 32 1 > datasets/test_1_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\n\t== Test 2 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 32 x 32 map \2048 x 3 vectors"
#./parallel_SOM datasets/input_2048_3_gauss.dat 32 1 > datasets/test_2_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\n\t== Test 3 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 32 x 32 map \t5120 x 1 vectors"
./parallel_SOM datasets/input_5120_1_gauss.dat 32 1 > datasets/test_3_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\n\t== Test 4 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 32 x 32 map \t5120 x 8 vectors"
#./parallel_SOM datasets/input_5120_8_gauss.dat 32 1 > datasets/test_4_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\n\t== Test 5 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 16 x 16 map \t5120 x 3 vectors"
./parallel_SOM datasets/input_5120_3_gauss.dat 16 1 > datasets/test_5_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\n\t== Test 6 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 48 x 48 map \t5120 x 3 vectors"
./parallel_SOM datasets/input_5120_3_gauss.dat 48 1 > datasets/test_6_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

./big_map_test_script.sh
