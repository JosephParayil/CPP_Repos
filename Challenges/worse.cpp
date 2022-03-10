#include <iostream>
using namespace std;

void problem() {
    string worse; getline(cin,worse);
    string decoded;
    int count=0;
    for (unsigned int i=0;i<worse.size();i++) {
        if (worse[i]==' ') {
            decoded.push_back(64+count);
            count=0;
        } else count++;
    }
    decoded.push_back(64+count);
    count=0;
    cout << decoded << '\n';
}

int main() {
    int t = 0; cin >> t;
    string in;
    getline(cin,in);
    while (t--) problem();
    return 0;
}