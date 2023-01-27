cp_distance = function(cp1, cp2){
  mx = 0
  for (c in cp1){
    mx = max(mx, min(abs(cp2 - c)))
  }
  for (c in cp2){
    mx = max(mx, min(abs(cp1 - c)))
  }
  return(mx)
}
