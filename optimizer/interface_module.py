class Base:
    def update(self, base, delta):
        pass
    
    def partialUpdate(self, base, delta, index, max_index):
        """업데이트를 부분적으로 수행합니다."""
        raise NotImplementedError
