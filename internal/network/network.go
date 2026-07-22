package network

type MLP struct {
	inner  Layer
	outer  Layer
	hidden []Layer
}

func NewMLP() *MLP {
	return &MLP{}
}
