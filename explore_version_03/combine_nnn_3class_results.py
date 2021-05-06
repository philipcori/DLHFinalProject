import csv
from explore_version_03.utils.measure import MeasureR

def main():
    results_path_nnn = './explore_version_03/results_whole_dataset_first_stage/resnet152_20200407_multiclass_cv5/result_detail_resnet152_test_cv5.csv'
    results_path_3classes = './explore_version_03/results/flannel_20200719_10_multiclass_cv5_focal/result_detail.csv'
    dest_results_dir = './explore_version_03/results_hierarchical'
    dest_results_dir_path = dest_results_dir + '/result_detail.csv'

    normal_i = 2

    results = []
    all_nnn_rows = []
    all_3classes_rows = []
    with open(results_path_nnn) as nnn_results_file:
        reader_nnn = csv.reader(nnn_results_file)
        for row in reader_nnn:
            all_nnn_rows.append([float(elt) for elt in row])

    with open(results_path_3classes) as results_file_3classes:
        reader_3classes = csv.reader(results_file_3classes)
        for row in reader_3classes:
            all_3classes_rows.append([float(elt) for elt in row])

    for (i, row_nnn) in enumerate(all_nnn_rows):
        row_3classes = all_3classes_rows[i]
        real_labels = row_nnn[2:]
        if row_nnn[0] > row_nnn[1]:     # predicted normal
            results.append([0.0,0.0,0.0,1.0] + real_labels)
        else:
            if max(row_3classes[:3]) == row_3classes[0]:
                results.append([1.0,0.0,0.0,0.0] + real_labels)
            elif max(row_3classes[:3]) == row_3classes[1]:
                results.append([0.0,1.0,0.0,0.0] + real_labels)
            else:
                results.append([0.0,0.0,1.0,0.0] + real_labels)

    with open(dest_results_dir_path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        for pred in results:
            writer.writerow(pred)

    mr = MeasureR(dest_results_dir, 0, 0)
    mr.output()


if __name__ == '__main__':
    main()