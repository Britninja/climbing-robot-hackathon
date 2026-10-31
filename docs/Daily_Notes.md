# Daily Notes

## Daily Notes - Day 1 (29th Oct 2026)

- short term action planning
- learning doing PPO
- long term planning where RL doesn't scale

Wolfenbüttel

- Ostfalia  University of Applied Sciences
- Reinhard Gernd
- Robotic Autonomy

## Daily Notes - Day 2 (30th Oct 2026)

Rules

- get into the socket of the hold (penetrate the plane of the cup => solid grip on the hole => has to be inside)
- time of hold doesn't matter

Level 1

- missing rows (since there is not enough holds)

Statics

- one example with minc, one with pincer (c backend instead of pniocchio)

- support polygon => not falling through the ground because we have a force from the ground
- (red) center of mass = area between the feet of feet => we cannot sustain a motion / keep standing up if the center of mass moves outside the support polygone
- friction negligible for the balls
- maximum force (max weight?) the hold can sustain (not considering the angle of the elbows for example (if your arm is straight you need little force, if you are bent you need more torque force))
- angles not fully taken into account (it's not the ground truth)
- if you are at the boundary, one motor is always working hard
- you can also leave the support polygon, but staying in it would be are sufficient (real climbers might launch off and swing)
- eg. try out disabling the left_hand => your support polygon shrinks, see how you can move the hand without extending the polygon
- alternative : task space inverse dynamics solver => also need to specify contacts
