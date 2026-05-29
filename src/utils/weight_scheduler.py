class WeightScheduler:
    """
    Weight scheduler to adjust weights over training steps.
    Supports linear, constant, and exponential increase.
    Inputs:
        type: str, one of ["linear", "constant", "exponential, "heaviside"]
        initial_weight: int, starting weight
        final_weight: int, final weight
        total_steps: int, number of steps over which to adjust the weight
    """
    def __init__(self, type, initial_weight: int, final_weight: int, total_steps: int):
        self.type = type
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.total_steps = total_steps

        # Sanity checks
        assert self.initial_weight < self.final_weight, "initial_weight must be less than final_weight"
        assert isinstance(self.initial_weight, int), "initial_weight must be an integer"
        assert isinstance(self.final_weight, int), "final_weight must be an integer"
        assert self.type in ["linear", "constant", "exponential", "heaviside"], "type must be one of ['linear', 'constant', 'exponential', 'heaviside']"

        # Current weight starts at initial_weight
        self.cur_weight = initial_weight

        # Internal counter for steps taken
        self._count_cur_step = 0

    def step(self):
        """Advance the weight scheduler by one step and return the current weight."""
        if self.type == "constant":
            return self.cur_weight

        elif self.type == "linear":
            if self.cur_weight < self.final_weight:
                self.cur_weight += (self.final_weight - self.initial_weight) / self.total_steps

        elif self.type == "exponential":
            if self._true_weight < self.final_weight:
                growth_rate = (self.final_weight / self.initial_weight) ** (1 / self.total_steps)
                self.cur_weight *= growth_rate

        elif self.type == "heaviside":
            if self._count_cur_step >= self.total_steps:
                self.cur_weight = self.final_weight

        if self.cur_weight > self.final_weight:
            self.cur_weight = self.final_weight
        self._count_cur_step += 1

        return self.cur_weight

    def get_weight(self):
        """Get the current weight without advancing the scheduler."""
        return self.cur_weight
