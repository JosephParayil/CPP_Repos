#include <iostream>
using namespace std;

void problem() {
    int n, now,pre; cin>> n >>pre;
    unsigned long long count = 0;
    n--;
    while(n--) {
        cin>>now;
        if (now<=pre) {
            count+= 1+pre-now;
            pre = pre+1;
        } else pre = now;
    }
    cout << count << '\n';
}

int main() { 
    int t = 0; cin >> t;
    while (t--) problem();
    return 0;
}