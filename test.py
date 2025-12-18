import math
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
import skfuzzy.control as ctrl


class InvertedPendulum:
	g = 9.81  # ускорение свободного падения

	def __init__(self, m, I, L, dT, init_th, init_w, delay):
		self.m = m  # масса тела, кг
		self.I = I  # момент инерции
		self.L = L  # расстояние до центра тяжести
		self.dT = dT  # шаг моделирования
		self.delay = delay  # задержка
		self.delay_shift = int(math.ceil(self.delay / self.dT))
		self.friction_coeff = 10  # коэффициент трения

		self.th = init_th  # угол
		self.w = init_w  # угловая скорость

		self.draw_plots = False
		self.draw_pendulum = False
		self.vxs = []
		self.vys = []
		self.cxs = []
		self.cys = []
		self.ths = []
		self.ws = []
		self.Ms = []  # добавим для отображения момента

	def enable_plots_drawing(self, enable):
		self.draw_plots = enable
		self.draw_pendulum = False

	def enable_pendulum_drawing(self, enable):
		self.draw_plots = False
		self.draw_pendulum = enable

	def _draw_plots(self):
		if self.draw_plots:
			plt.clf()
			plt.plot(self.vxs, label='vx')
			plt.plot(self.vys, label='vy')
			plt.plot(self.ths, label='θ')
			plt.plot(self.ws, label='ω')
			plt.plot(np.array(self.Ms), label='M')

			plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
			plt.grid(True)
			plt.tight_layout()

			plt.show(block=False)
			plt.pause(0.001)

	def _draw_pendulum(self):
		if self.draw_pendulum:
			plt.clf()
			cx = -self.L * math.sin(self.th)
			cy = self.L * math.cos(self.th)
			plt.plot([-1.5 * self.L, 1.5 * self.L], [0, 0], 'k-')
			plt.plot([0, 0], [-1.5 * self.L, 1.5 * self.L], 'k-')
			plt.plot([0, cx], [0, cy], 'b-o')
			plt.title('Положение маятника')
			plt.show(block=False)
			plt.pause(0.001)

	def step(self, M):
		# Компоненты скорости
		vx = -self.L * self.w * math.cos(self.th)
		vy = -self.L * self.w * math.sin(self.th)

		# Координаты
		cx = -self.L * math.sin(self.th)
		cy = self.L * math.cos(self.th)

		M_gravity = self.m * self.g * self.L * math.sin(self.th)

		# Момент силы трения
		M_friction = -self.friction_coeff * self.w

		# Суммарный момент
		M_total = M + M_gravity + M_friction

		# Угловое ускорение
		alpha = M_total / self.I

		# Обновление состояния
		self.w += alpha * self.dT
		self.th += self.w * self.dT

		# Сохраняем все переменные
		self.vxs.append(vx)
		self.vys.append(vy)
		self.cxs.append(cx)
		self.cys.append(cy)
		self.ths.append(self.th)
		self.ws.append(self.w)
		self.Ms.append(M)

		# Визуализация
		self._draw_plots()
		self._draw_pendulum()

		# Возврат состояния с задержкой
		out_index = max(-1 - self.delay_shift, -len(self.ths))
		return (self.ths[out_index], self.ws[out_index])


class FuzzyRegulator:
	def __init__(self):
		# Фазификация входных переменных
		range_th = np.linspace(-math.pi, math.pi, 5)
		range_w = np.linspace(-math.pi, math.pi, 5)
		range_M = np.linspace(-10., 10., 5)

		fuzzy_th = ctrl.Antecedent(range_th, 'th')
		fuzzy_w = ctrl.Antecedent(range_w, 'w')
		fuzzy_M = ctrl.Consequent(range_M, 'M')

		fuzzy_th.automf(names=['low', 'medium', 'high'])
		fuzzy_w.automf(names=['low', 'medium', 'high'])
		fuzzy_M.automf(names=['low', 'medium', 'high'])

		# r1: Если отклонение сильное в одну сторону -> даём сильный момент в противоположную
		r1 = ctrl.Rule(antecedent=((fuzzy_th['low'] & fuzzy_w['low']) |
								   (fuzzy_th['low'] & fuzzy_w['medium']) |
								   (fuzzy_th['medium'] & fuzzy_w['low'])),
					   consequent=fuzzy_M['high'], label='rule high')

		# r2: Если отклонение сильное в другую сторону -> даём сильный момент в противоположную
		r2 = ctrl.Rule(antecedent=((fuzzy_th['high'] & fuzzy_w['high']) |
								   (fuzzy_th['high'] & fuzzy_w['medium']) |
								   (fuzzy_th['medium'] & fuzzy_w['high'])),
					   consequent=fuzzy_M['low'], label='rule low')

		system = ctrl.ControlSystem(rules=[r1, r2])
		self.sim = ctrl.ControlSystemSimulation(system)

	def get_M(self, state):
		th, w = state
		# Ограничиваем входные значения
		th_clipped = np.clip(th, -math.pi, math.pi)
		w_clipped = np.clip(w, -math.pi, math.pi)

		self.sim.input['th'] = th_clipped
		self.sim.input['w'] = w_clipped

		try:
			self.sim.compute()
			M_output = self.sim.output['M']

			return M_output * 0.1

		except Exception as e:
			print(f"Ошибка при вычислении момента: {e}")
			return 0.0


def model():
	dT = 0.01
	modelling_time = 60.0

	m = 0.3
	I = 0.1
	L = 0.1
	delay = 0.3  # 0.1 или 0.3

	init_th = 0.2
	init_w = 0.0

	pnd = InvertedPendulum(m, I, L, dT, init_th, init_w, delay)
	pnd.enable_plots_drawing(True)
	pnd.enable_pendulum_drawing(1)

	reg = FuzzyRegulator()

	curr_t = 0.
	curr_M = 0.
	while curr_t < modelling_time:
		state = pnd.step(curr_M)
		curr_M = reg.get_M(state)
		curr_t += dT


model()