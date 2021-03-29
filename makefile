install:
	unzip -a ./transactions.txt.zip
	pip3 install -r requirements.txt --no-index

#Code for Question 1
run_pandas_profiling:
	python3 ./code/Q1_profiling.py

#Code for Question 1 and 2
run_EDA:
	python3 ./code/Q1_Q2_EDA.py

#Code for Question 3
find_duplicate_transactions:
	python3 ./code/Q3_find_duplicates.py

#Code for Question 4
build_model:
	python3 ./code/Q4_build_model.py