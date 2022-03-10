#include <iostream>
using namespace std;

void problem() {
    int x,y; cin >> x >> y;
    cout << (x+y)*2 << '\n';
}

int main() {
    int t = 0; cin >> t;
    while (t--) problem();
    return 0;
}