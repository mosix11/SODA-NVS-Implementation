from src.models.noise_schedules import get_schedule
from visualization import plot_noise_schedules
T = 1000
cosine_sche = get_schedule('cosine', T, type='DDPM')
inverted_cosine_sche = get_schedule('inverted', T, type='DDPM')
for k,v in cosine_sche.items():
    print(k, v.shape)
plot_noise_schedules(cosine_sche, inverted_cosine_sche, T)