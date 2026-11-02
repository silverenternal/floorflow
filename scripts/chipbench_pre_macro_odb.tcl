# Tcl equivalent of ChiPBench pre_macro.py for images without OpenROAD Python.
read_db $::env(INPUT_ODB)
set block [ord::get_db_block]
set core [$block getCoreArea]
set die [$block getDieArea]
set core_xmin [$core xMin]
set core_ymin [$core yMin]
set core_xmax [$core xMax]
set core_ymax [$core yMax]
set center_x [expr {($core_xmin + $core_xmax) / 2}]
set center_y [expr {($core_ymin + $core_ymax) / 2}]

set macro_xmin 2147483647
set macro_ymin 2147483647
set macro_xmax -2147483648
set macro_ymax -2147483648
set macro_count 0
foreach inst [$block getInsts] {
  set master [$inst getMaster]
  if {[$master getType] eq "BLOCK"} {
    $inst setPlacementStatus "PLACED"
    $inst setPlacementStatus "FIRM"
    set bbox [$inst getBBox]
    set x0 [$bbox xMin]
    set y0 [$bbox yMin]
    set x1 [$bbox xMax]
    set y1 [$bbox yMax]
    if {$x0 < $macro_xmin} {set macro_xmin $x0}
    if {$y0 < $macro_ymin} {set macro_ymin $y0}
    if {$x1 > $macro_xmax} {set macro_xmax $x1}
    if {$y1 > $macro_ymax} {set macro_ymax $y1}
    incr macro_count
  } else {
    $inst setLocation $center_x $center_y
    $inst setOrient "R0"
  }
}

if {$macro_count == 0} {
  error "No block-level macros found in input ODB"
}
if {$macro_xmin < [$die xMin] || $macro_ymin < [$die yMin] ||
    $macro_xmax > [$die xMax] || $macro_ymax > [$die yMax]} {
  error "Macro bounding box lies outside die area"
}
write_db $::env(OUTPUT_ODB)
