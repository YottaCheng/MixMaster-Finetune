1.	generate_wordcloud.py 根据/Volumes/Study/prj/data/raw/training_raw_data.docx生成词云前20个/Volumes/Study/prj/data/processed/top_20_words.json生成这个文件
2.	然后renew_json.py 将前20个词中写出相关的替换词词典到/Volumes/Study/prj/config/music_synonyms.json
3.	/Volumes/Study/prj/scripts/pure_eda.py 将/Volumes/Study/prj/data/raw/training_raw_data.docx进行eda替换 生成到/Volumes/Study/prj/data/processed/augmented_results.docx
4.	backtracking.py 将/Volumes/Study/prj/data/raw/training_raw_data.docx进行backtracking和反义词替代/Volumes/Study/prj/data/processed/backtrans_results.docx
5.	/Volumes/Study/prj/scripts/filter.py将/Volumes/Study/prj/data/processed/augmented_results.docx和/Volumes/Study/prj/data/processed/backtrans_results.docx进行初步筛选和清洗 生成/Volumes/Study/prj/data/processed/filtered_results.docx
6.	/Volumes/Study/prj/scripts/prelabeled.py将/Volumes/Study/prj/data/processed/filtered_results.docx进行预标注 生成到/Volumes/Study/prj/data/raw/training_labeled_data.docx
且中途有会生成随机的抽查 检查7个标签 是否符合标准
7.	/Volumes/Study/prj/scripts/clean_data.py将/Volumes/Study/prj/data/raw/training_labeled_data.docx再一步进行清洗 然后标号 降低信息熵 为训练做完所有的准备
