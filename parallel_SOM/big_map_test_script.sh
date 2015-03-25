echo "== TEST SCRIPT FOR PARALLEL SOM: BIG MAPS =="
echo -e "\t== Test 1 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 32 x 32 map \t5120 x 3 vectors"
# ./parallel_SOM datasets/input_5120_3_gauss.dat 32 1 > datasets/test_1_big_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\t== Test 2 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 64 x 64 map \t5120 x 3 vectors"
./parallel_SOM datasets/input_5120_3_gauss.dat 64 1 > datasets/test_2_big_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\t== Test 3 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 96 x 96 map \t5120 x 3 vectors"
./parallel_SOM datasets/input_5120_3_gauss.dat 96 1 > datasets/test_3_big_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW

echo -e "\t== Test 4 =="
NOW=`date`
echo -e "\tStart\t" $NOW
echo -e "\tRunning 128 x 128 map \t5120 x 3 vectors"
./parallel_SOM datasets/input_5120_3_gauss.dat 128 1 > datasets/test_4_big_output.txt
NOW=`date`
echo -e "\tEnd\t" $NOW