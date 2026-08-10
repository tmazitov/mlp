package vector

func (v Vector[K]) AngleCos(u Vector[K]) float64 {
	return float64(v.Dot(u)) / (v.Norm() * u.Norm())
}
