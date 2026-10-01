NAME    := mlp
ANALYZE := analyze

GO   := go
SRCS := $(shell find . -name '*.go')

.PHONY: all clean fclean re

all: $(NAME) $(ANALYZE)

$(NAME): $(SRCS)
	$(GO) mod download
	$(GO) build -o $(NAME) ./cmd

$(ANALYZE): $(SRCS)
	$(GO) build -o $(ANALYZE) ./cmd/analyze

clean:
	$(GO) clean

fclean: clean
	$(RM) $(NAME) $(ANALYZE)

re: fclean all
