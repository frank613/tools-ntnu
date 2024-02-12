import sys
import pdb

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("this script takes 2 arguments <GOP file> <error-uttid-list>")
	with open(sys.argv[2], 'r') as inF:
		error_list = []
		for line in inF: 	
			fields = line.strip().split()
			if len(fields) != 1:
				sys.exit("wrong input line")
			error_list.append(fields[0])
	with open(sys.argv[1], 'r') as inF:
		printing = False
		for line in inF:
			fields = line.strip().split()
			if len(fields) == 1 and fields[0] not in error_list:
				printing = True
				print(" ".join(fields))
				continue
			if len(fields) == 0:
				if printing:
					print("")
				printing = False
				continue
			if printing:
				print(" ".join(fields))


	
