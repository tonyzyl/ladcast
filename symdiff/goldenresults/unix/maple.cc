#include <cmath>
using namespace std;
void fs(const double a, const double b, double &f1, double &f2, double &f3)
{
const double unique0 = sin(a);
const double unique2 = cos(a);
const double unique9 = (unique0 * unique2);
f1 = ((b * unique0) + unique9);
const double unique12 = (b * unique0 * unique2);
const double unique6 = cos(b);
f2 = ((unique0 * unique2 * unique6) + unique12 + unique9);
f3 = (-unique12 + (a * unique2));
}
