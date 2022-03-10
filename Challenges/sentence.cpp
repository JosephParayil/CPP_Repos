#include <iostream>
using namespace std;

int main() {
    int t = 0; cin >> t;
    for (int g=1;g<=t;g++) {
        int n; cin>> n;
        string input;
        int cap=0;
        int low=0;
        while(n--) {
            cin >> input;
            for (int i=0;i<input.size();i++) {
                if (input[i]>64&&91>input[i]) cap++;
                else low++;
            }
        }
        cout << "Sentence #" << g << ((cap==low) ? ": Yes\n" : ": No\n");
    }
    return 0;
}