for app in surf surf_lfsr sift sift_lfsr
do
    for num in 4 20 50 100
    do
        echo "$app -- $num"
        ruby test.rb $num $app
    done
done
