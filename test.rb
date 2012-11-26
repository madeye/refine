io = 0.0
compute = 0.0
for i in 1..100
    time = `./refine #{i} #{ARGV[0]} /home/mmrb/ukbench/temp/results_#{ARGV[1]} /home/mmrb/ukbench/points_#{ARGV[1]}`
    times = time.split(",")
    compute += times[0].to_f
    io += times[1].to_f
    puts time
end
puts format("%.2f %.2f", io/100.0, compute/100.0)
puts format("%.2f", compute / (io + compute))
