class AnalyzerResult:
    def __init__(
        self,
        entity_type: str,
        start: int,
        end: int,
        score: float,
        text: str = ""
    ):
        self.entity_type = entity_type
        self.start = start
        self.end = end
        self.score = score
        self.text = text