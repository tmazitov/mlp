package vector

func (v Vector[K]) AngleCos(u Vector[K]) float32 {
	return float32(v.Dot(u)) / (v.Norm() * u.Norm())
}
