__kernel void bitonic_merge_int(__global int *arr, int sub_arr_size, int step) {

	int i = get_global_id (0);

	if (i % (step * 2) < step) {

		int ascend = 1;

		if ((i / sub_arr_size) % 2 == 1)
			ascend = 0;

		int first  = arr[i];
		int second = arr[i + step];

		if ((first > second) && ascend) {

			arr[i] = second;
			arr[i + step] = first;
		}
		else if ((first < second) && !ascend) {

			arr[i] = second;
			arr[i + step] = first;
		}
	}
}
