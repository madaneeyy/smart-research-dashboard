import { LandingPage } from "./components/LandingPage";

function App() {
  const handleTryDemo = () => {
    console.log("Demo clicked");
  };

  const handleSignIn = () => {
    console.log("Sign in clicked");
  };

  return (
    <LandingPage
      onTryDemo={handleTryDemo}
      onSignIn={handleSignIn}
    />
  );
}

export default App;