# Export the pre-macro-placement graph needed by the external DEF adapter.
set block [ord::get_db_block]
if {$block eq "NULL"} {
  error "No design block is open"
}
set database [ord::get_db]
set tech [$database getTech]

proc tab_clean {value} {
  return [string map [list "\t" " " "\n" " "] $value]
}

set die [$block getDieArea]
set core [$block getCoreArea]
puts "ORFS_DBU_PER_MICRON\t[$tech getDbUnitsPerMicron]"
puts "ORFS_DIE\t[$die xMin]\t[$die yMin]\t[$die xMax]\t[$die yMax]"
puts "ORFS_CORE\t[$core xMin]\t[$core yMin]\t[$core xMax]\t[$core yMax]"

foreach inst [$block getInsts] {
  set master [$inst getMaster]
  if {[$master getType] eq "BLOCK"} {
    lassign [$inst getLocation] location_x location_y
    puts [join [list \
      ORFS_MACRO \
      [tab_clean [$inst getName]] \
      [tab_clean [$master getName]] \
      [$master getWidth] \
      [$master getHeight] \
      $location_x \
      $location_y \
      [$inst getPlacementStatus] \
      [$inst getOrient] \
    ] "\t"]
  }
}

foreach bterm [$block getBTerms] {
  set box [$bterm getBBox]
  if {$box eq "NULL"} {
    continue
  }
  set x [expr {([$box xMin] + [$box xMax]) / 2.0}]
  set y [expr {([$box yMin] + [$box yMax]) / 2.0}]
  puts [join [list ORFS_PORT [tab_clean [$bterm getName]] $x $y] "\t"]
}

foreach net [$block getNets] {
  set net_name [tab_clean [$net getName]]
  foreach iterm [$net getITerms] {
    set inst [$iterm getInst]
    if {[[$inst getMaster] getType] eq "BLOCK"} {
      puts [join [list \
        ORFS_NET_ITERM \
        $net_name \
        [tab_clean [$inst getName]] \
        [tab_clean [[$iterm getMTerm] getName]] \
      ] "\t"]
    }
  }
  foreach bterm [$net getBTerms] {
    puts [join [list \
      ORFS_NET_BTERM \
      $net_name \
      [tab_clean [$bterm getName]] \
    ] "\t"]
  }
}
