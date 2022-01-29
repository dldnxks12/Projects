function H = get_CC(th2, th3, dth1, dth2, dth3)

global lz1 lz2 lz3 L1 L2 L3 g m1 m2 m3 r1 r2 r3 tau1 tau2 tau3;

H = [(- L1*m3*(r3*sin(th2 + th3) + L2*sin(th2)) - L1*m2*r2*sin(th2))*dth2^2 - 2*m3*r3*(L1*sin(th2 + th3) + L2*sin(th3))*dth2*dth3 - 2*dth1*(L1*m3*(r3*sin(th2 + th3) + L2*sin(th2)) + L1*m2*r2*sin(th2))*dth2 - m3*r3*(L1*sin(th2 + th3) + L2*sin(th3))*dth3^2 - 2*dth1*m3*r3*(L1*sin(th2 + th3) + L2*sin(th3))*dth3;
                                                                                                                                                (L1*m3*(r3*sin(th2 + th3) + L2*sin(th2)) + L1*m2*r2*sin(th2))*dth1^2 - 2*L2*m3*r3*sin(th3)*dth1*dth3 - L2*m3*r3*sin(th3)*dth3^2 - 2*L2*dth2*m3*r3*sin(th3)*dth3;
                                                                                                                                                                                                      m3*r3*(L1*sin(th2 + th3) + L2*sin(th3))*dth1^2 + 2*L2*m3*r3*sin(th3)*dth1*dth2 + L2*m3*r3*sin(th3)*dth2^2];