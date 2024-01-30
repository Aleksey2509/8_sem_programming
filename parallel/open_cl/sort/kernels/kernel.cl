__kernel void bitonic_merge_int(__global int *sequence, int subSeqSize, int step) {

	int i = get_global_id (0);

	if (i % (step * 2) < step) {

		int upORdown = 1;

		if ((i / subSeqSize) % 2 == 1)
			upORdown = 0;

		int first  = sequence[i];
		int second = sequence[i + step];

		if ((first > second) && upORdown) {

			sequence[i] = second;
			sequence[i + step] = first;
		}
		else if ((first < second) && !upORdown) {

			sequence[i] = second;
			sequence[i + step] = first;
		}
	}
}
