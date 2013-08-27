class Scheduler:
  @staticmethod
  def makeScheduler(alg, trainer):
    if alg == 'smooth':
      return SmoothScheduler(trainer)

    if alg == 'increment':
      return IncrementScheduler(trainer)

  def __init__(self, trainer):
    self.trainer = trainer

  def should_continue_training(self):
    return True

  def check_test_data(self):
    return True

  def check_save_checkpoint(self):
    return True

  def reset(self):
    pass

class SmoothScheduler(Scheduler):
  def __init__(self, trainer):
    Scheduler.__init__(self, trainer)
    self.step_level = [5, 10, 12, 15]
    self.test_accu = []
    self.step = self.step_level[0]
    self.prev_avg = 0.0
    self.num_test_outputs = 0
    self.keep = True
    assert self.trainer.test_freq == self.trainer.save_freq, 'the save freq must be equal to test freq'
    assert len(self.trainer.test_range) == 1, 'the test set could only have one batch'

  def should_continue_training(self):
    return self.keep

  def check_save_checkpoint(self):
    if self.trainer.test_outputs == []:
      return True
    num = len(self.trainer.test_outputs)
    if num != self.num_test_outputs:
      self.num_test_outputs = num
      self.test_accu.append(1 - self.trainer.test_outputs[-1][0]['logprob'][1])
      if len(self.test_accu) <= self.step:
        self.keep = True
      else:
        avg = sum(self.test_accu[-(1 + self.step) : -1]) / self.step

        if avg < self.prev_avg:
          self.keep = False
        else:
          self.keep = True
          self.prev_avg = avg
    return self.keep

  def reset(self):
    self.test_accu = []
    self.keep = True
    self.prev_avg = 0.0
    self.step = self.step_level[0]
    self.num_test_outputs = 0

  def set_level(self, level):
    if level >= len(self.step_level):
      self.step = self.step_level[-1]
    else:
      self.step = self.step_level[level]
