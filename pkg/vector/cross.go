package vector

func CrossProduct[K numberType](v, u Vector[K]) Vector[K] {

	if len(u) != 3 || len(v) != 3 {
		return nil
	}

	i := v[1]*u[2] - u[1]*v[2]
	j := (v[0]*u[2] - u[0]*v[2]) * -1
	k := v[0]*u[1] - u[0]*v[1]

	return Vector[K]{i, j, k}
}
