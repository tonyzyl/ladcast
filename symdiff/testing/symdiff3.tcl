package require symdifftcl
# Any copyright is dedicated to the Public Domain.
# http://creativecommons.org/publicdomain/zero/1.0/

symdiff "define_model(a, b + d)"
symdiff "define_model(c, b + d)"
puts "\n\nBEFORE\n\n"
foreach {i} [model_list] {
  puts [format "%s, %s" $i [symdiff "model_value($i)"] ]
}

symdiff "define_model(x, a + c)"

subexpression
puts "\n\nAFTER\n\n"
foreach {i} [model_list] {
  puts [format "%s, %s" $i [symdiff "model_value($i)"] ]
}

puts "\n\nORDERED for a c\n\n"
foreach {i} [ordered_list a c] {
  puts [format "%s, %s" $i [symdiff "model_value($i)"] ]
}

puts "\n\nORDERED for x\n\n"
foreach {i} [ordered_list x] {
  puts [format "%s, %s" $i [symdiff "model_value($i)"] ]
}

puts "\n\nORDERED for a\n\n"
foreach {i} [ordered_list a] {
  puts [format "%s, %s" $i [symdiff "model_value($i)"] ]
}
