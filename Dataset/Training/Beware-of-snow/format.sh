for i in Training/*;
do 
	for j in $i/*;
	do
		mogrify -format ppm *.jpg
	done 
done