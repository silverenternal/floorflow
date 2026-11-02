# Count block-level instances in an OpenROAD database without changing it.
set block [ord::get_db_block]
if {$block eq "NULL"} {
  error "No design block is open"
}

set macro_count 0
set macro_masters {}
foreach inst [$block getInsts] {
  set master [$inst getMaster]
  if {[$master getType] eq "BLOCK"} {
    incr macro_count
    lappend macro_masters [$master getName]
  }
}

puts "ORFS_MACRO_INSTANCE_COUNT\t$macro_count"
foreach master_name [lsort -unique $macro_masters] {
  puts "ORFS_MACRO_MASTER\t$master_name"
}
